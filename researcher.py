"""
Autonomous research loop for power spectrum emulator.
Calls Claude API to propose hyperparameter changes, commits them to git,
runs 5-minute training trials, and loops indefinitely — keeping improvements,
reverting failures.

Inspired by: https://github.com/karpathy/autoresearch/blob/master/program.md
"""

import ast
import os
import re
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKDIR      = Path('/global/cfs/cdirs/desicollab/users/epaillas/code/trainloop')
TRAIN_SCRIPT = WORKDIR / 'train_loop.py'
RESULTS_TSV  = WORKDIR / 'results.tsv'
RUN_LOG      = WORKDIR / 'run.log'
MODEL_NAME   = 'claude-opus-4-6'
TRAINING_TIMEOUT    = 600   # 10 min wall-clock kill threshold (seconds)
MAX_CONTEXT_TURNS   = 20    # prune conversation history beyond this

GIT_REMOTE = 'git@github.com:epaillas/train-loop.git'

# Branch name uses today's date, e.g. autoresearch/mar9
_now = datetime.now()
GIT_BRANCH = f'autoresearch/{_now.strftime("%b").lower()}{_now.day}'

# ---------------------------------------------------------------------------
# Baseline code (fixed train_loop.py with get_hyperparams + markers)
# ---------------------------------------------------------------------------

BASELINE_CODE = '''\
import numpy as np
from datetime import timedelta
from pathlib import Path
from sunbird.emulators import FCN
from sunbird.emulators.train import FCNTrainer
from sunbird.data import ArrayDataModule
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
import torch
import argparse

torch.set_float32_matmul_precision(\'high\')

def _build_transform(transform_name):
    if transform_name is None:
        return None
    if transform_name == \'log\':
        return LogTransform()
    if transform_name == \'arcsinh\':
        return ArcsinhTransform()
    raise ValueError(f\'Unknown transform: {transform_name}\')


def load_data(observable_name):
    data_dir = Path(\'/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.3/abacus/compressed\')
    data = np.load(data_dir / f\'{observable_name}.npy\', allow_pickle=True).item()
    x = data[\'x\'][\'data\'] # shape (ncosmo, nhod, nparams)
    y = data[\'y\'][\'data\'] # shape (ncosmo, nhod, npoles, nk)

    # select first 6 cosmologies for testing, rest for training/validation
    x_train = x[6:, :]
    y_train = y[6:, :]

    # now reshape to (nsamples, n_features)
    x_train = x_train.reshape(-1, x_train.shape[-1])
    y_train = y_train.reshape(-1, y_train.shape[-2] * y_train.shape[-1])

    print(\'Loaded train x, y with shapes:\', x_train.shape, y_train.shape)

    covariance_y = data[\'covariance_y\'][\'data\']
    covariance_y = covariance_y.reshape(covariance_y.shape[0], -1)
    covariance_matrix = np.cov(covariance_y, rowvar=False)
    print(\'Loaded covariance matrix with shape:\', covariance_matrix.shape)

    return x_train, y_train, covariance_matrix


# BEGIN MODIFIABLE
def get_hyperparams():
    """Tunable hyperparameters. Valid options documented in docstring.

    act_fn options: learned_sigmoid, SiLU, GELU, ELU, Mish, Tanh, ReLU, LeakyReLU, SELU, Hardswish
    loss options: weighted_mae, weighted_mse, GaussianNLoglike, mse, rmse, mae
    transform_input/output: None or \'arcsinh\' (log is INVALID: inputs/multipoles can be negative)
    """
    return dict(
        learning_rate=1e-3,
        n_hidden=[512, 512, 512, 512],
        dropout_rate=0.0,
        weight_decay=0,
        batch_size=128,
        val_fraction=0.1,
        act_fn=\'learned_sigmoid\',        # or: SiLU, GELU, ELU, Mish, ReLU, etc.
        loss=\'weighted_mae\',             # or: weighted_mse, GaussianNLoglike, mse, mae
        transform_input=None,            # or: \'arcsinh\' (log is invalid: inputs can be negative)
        transform_output=None,           # or: \'arcsinh\' (log is INVALID: multipoles can be negative at large k)
        scheduler_patience=10,
        scheduler_factor=0.5,
        scheduler_threshold=1e-6,
        gradient_clip_val=0.5,
    )
# END MODIFIABLE


def TrainFCN(x, y, covariance_matrix, learning_rate, n_hidden, dropout_rate,
             weight_decay, batch_size=128, val_fraction=0.1, act_fn=\'learned_sigmoid\',
             loss=\'weighted_mae\', transform_input=None, transform_output=None,
             scheduler_patience=10, scheduler_factor=0.5, scheduler_threshold=1e-6,
             gradient_clip_val=0.5, seed=None, max_time_minutes=5):

    np.random.seed(seed)

    input_transform = _build_transform(transform_input)
    output_transform = _build_transform(transform_output)

    if input_transform is not None:
        x = input_transform.transform(x)

    if output_transform is not None:
        y = output_transform.transform(y)

    train_mean = np.mean(y, axis=0)
    train_std = np.std(y, axis=0)

    train_mean_x = np.mean(x, axis=0)
    train_std_x = np.std(x, axis=0)

    data = ArrayDataModule(
        x=torch.Tensor(x),
        y=torch.Tensor(y),
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=0
    )
    data.setup()

    model = FCN(
        n_input=data.n_input,
        n_output=data.n_output,
        n_hidden=n_hidden,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        scheduler_threshold=scheduler_threshold,
        weight_decay=weight_decay,
        act_fn=act_fn,
        loss=loss,
        training=True,
        mean_output=train_mean,
        std_output=train_std,
        mean_input=train_mean_x,
        std_input=train_std_x,
        transform_input=input_transform,
        transform_output=output_transform,
        standarize_output=True,
        covariance_matrix=covariance_matrix,
    )

    dump_dir = \'/pscratch/sd/e/epaillas/dump/train\'
    trainer = FCNTrainer(
        checkpoint_dir=dump_dir,
        max_epochs=5000,
        devices=1,
        max_time=timedelta(minutes=max_time_minutes),
        logger=\'tensorboard\',
        log_dir=dump_dir,
        gradient_clip_val=gradient_clip_val,
    )
    val_loss = trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    return val_loss


if __name__ == \'__main__\':

    parser = argparse.ArgumentParser(description=\'Train FCN for EMC observables.\')
    parser.add_argument(\'-s\', \'--statistic\', type=str, default=\'spectrum\', help=\'Statistic to train on.\')
    parser.add_argument(\'--seed\', type=int, default=42, help=\'Random seed for reproducibility.\')
    parser.add_argument(\'--max_time_minutes\', type=float, default=5, help=\'Maximum training time in minutes. Defaults to 5.\')
    args = parser.parse_args()

    x, y, covariance_matrix = load_data(args.statistic)

    hp = get_hyperparams()
    val_loss = TrainFCN(x, y, covariance_matrix, seed=args.seed,
                        max_time_minutes=args.max_time_minutes, **hp)
    print(f"val_loss: {val_loss:.6f}")
'''

# ---------------------------------------------------------------------------
# System prompt for Claude
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert machine learning researcher helping to minimize the validation loss \
of a neural network emulator for cosmological power spectrum multipoles.

## Your Task
Propose changes to the `get_hyperparams()` function in `train_loop.py` to minimize `val_loss`. \
The baseline val_loss is approximately 0.189. Each experiment runs for 5 minutes.

## What You CAN Change
Only the return dict(...) body of `get_hyperparams()`. You MUST output the complete \
`# BEGIN MODIFIABLE ... # END MODIFIABLE` block including the function definition.

## What You CANNOT Change
- Imports
- `load_data()`
- `_build_transform()`
- `TrainFCN()`
- `__main__` block
- File paths

## Valid Parameter Values

### act_fn (activation function)
learned_sigmoid, SiLU, GELU, ELU, Mish, Tanh, ReLU, LeakyReLU, SELU, Hardswish

### loss function
weighted_mae, weighted_mse, GaussianNLoglike, mse, rmse, mae
DO NOT use: learned_gaussian, multivariate_learned_gaussian (they will crash)

### Transforms
- transform_input: None or 'arcsinh'
- transform_output: None or 'arcsinh'
- CRITICAL: 'log' is INVALID for both input and output — power spectrum multipoles \
can be negative at large k, so log transform will produce NaN. Only 'arcsinh' is safe \
for signed data.

## Strategy
- Build on what worked; try a clearly different direction after two consecutive failures
- Small targeted changes are often better than sweeping multiple changes at once
- Consider: learning rate (try 3e-4, 1e-4), architecture depth/width, regularization \
(weight_decay, dropout), batch size, scheduler settings, activation functions
- gradient_clip_val: try values between 0.1 and 2.0

## Output Format (REQUIRED — follow exactly)
Your response must contain exactly:

DESCRIPTION: <one sentence describing what you changed and why>
CODE:
# BEGIN MODIFIABLE
def get_hyperparams():
    \"\"\"...(docstring)...\"\"\"
    return dict(
        ...
    )
# END MODIFIABLE

Do not include any other code blocks. Do not modify anything outside the markers.
"""

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run(cmd, cwd=None, check=True, capture=True):
    """Run a shell command and return CompletedProcess."""
    return subprocess.run(
        cmd, shell=True, cwd=cwd or WORKDIR,
        capture_output=capture, text=True, check=check
    )


def setup_git_repo():
    """Initialise git repo, set remote, create and push autoresearch branch."""
    git_dir = WORKDIR / '.git'
    if not git_dir.exists():
        print('[git] Initialising new git repo ...')
        _run('git init')
        _run('git add -A')
        _run('git commit -m "initial commit"')
    else:
        print('[git] Git repo already exists.')

    # Ensure remote is set
    result = _run('git remote get-url origin', check=False)
    if result.returncode != 0:
        print(f'[git] Adding remote origin → {GIT_REMOTE}')
        _run(f'git remote add origin {GIT_REMOTE}')
    else:
        existing = result.stdout.strip()
        if existing != GIT_REMOTE:
            print(f'[git] Updating remote origin → {GIT_REMOTE}')
            _run(f'git remote set-url origin {GIT_REMOTE}')

    # Create/switch to autoresearch branch
    result = _run(f'git checkout -b {GIT_BRANCH}', check=False)
    if result.returncode != 0:
        # Branch already exists — switch to it
        _run(f'git checkout {GIT_BRANCH}')

    print(f'[git] On branch {GIT_BRANCH}')


def git_commit(message):
    """Stage train_loop.py, commit, push, and return short commit hash."""
    _run('git add train_loop.py')
    _run(f'git commit -m "{message}"')
    result = _run('git rev-parse --short HEAD')
    commit_hash = result.stdout.strip()
    # Push (best-effort; don't crash the loop if remote is unreachable)
    push = _run(f'git push -u origin {GIT_BRANCH}', check=False)
    if push.returncode != 0:
        print(f'[git] Warning: push failed — {push.stderr.strip()[:200]}')
    return commit_hash


def git_revert_one():
    """Hard-reset to HEAD~1 and force-push."""
    _run('git reset --hard HEAD~1')
    push = _run(f'git push --force origin {GIT_BRANCH}', check=False)
    if push.returncode != 0:
        print(f'[git] Warning: force-push after revert failed — {push.stderr.strip()[:200]}')


# ---------------------------------------------------------------------------
# Code-change helpers
# ---------------------------------------------------------------------------

MARKER_RE = re.compile(
    r'# BEGIN MODIFIABLE\n.*?# END MODIFIABLE',
    re.DOTALL
)


def apply_baseline_fixes():
    """Write the fixed baseline train_loop.py and commit it."""
    print('[setup] Writing baseline train_loop.py ...')
    TRAIN_SCRIPT.write_text(BASELINE_CODE)
    commit_hash = git_commit('baseline: fix train_loop.py bugs and add get_hyperparams()')
    print(f'[setup] Baseline committed as {commit_hash}')
    return commit_hash


def apply_code_change(new_block):
    """Replace the # BEGIN MODIFIABLE ... # END MODIFIABLE block in train_loop.py."""
    current = TRAIN_SCRIPT.read_text()
    if not MARKER_RE.search(current):
        raise ValueError('Could not find # BEGIN MODIFIABLE markers in train_loop.py')
    updated = MARKER_RE.sub(new_block, current, count=1)
    TRAIN_SCRIPT.write_text(updated)


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_training():
    """
    Run train_loop.py, capturing stdout/stderr to RUN_LOG.
    Returns (success: bool, timed_out: bool).
    """
    print('[train] Starting training run ...')
    try:
        with open(RUN_LOG, 'w') as log_file:
            proc = subprocess.Popen(
                [sys.executable, str(TRAIN_SCRIPT)],
                cwd=WORKDIR,
                stdout=log_file,
                stderr=log_file,
            )
        proc.wait(timeout=TRAINING_TIMEOUT)
        success = proc.returncode == 0
        return success, False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False, True


def parse_val_loss(log_path=None):
    """
    Parse val_loss from run log.
    Looks for the last line matching 'val_loss: <float>'.
    Returns float or None.
    """
    path = log_path or RUN_LOG
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return None
    matches = re.findall(r'^val_loss:\s*([0-9]+\.[0-9]+)', text, re.MULTILINE)
    if matches:
        return float(matches[-1])
    return None


# ---------------------------------------------------------------------------
# Results recording
# ---------------------------------------------------------------------------

def record_result(commit, val_loss, status, description):
    """Append a row to results.tsv."""
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text('commit\tval_loss\tstatus\tdescription\n')
    val_str = f'{val_loss:.6f}' if val_loss is not None else 'N/A'
    with open(RESULTS_TSV, 'a') as f:
        f.write(f'{commit}\t{val_str}\t{status}\t{description}\n')
    print(f'[result] commit={commit} val_loss={val_str} status={status} | {description}')


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def prune_conversation_history(history):
    """Keep the first 2 messages plus the last MAX_CONTEXT_TURNS*3 messages."""
    max_tail = MAX_CONTEXT_TURNS * 3
    if len(history) <= 2 + max_tail:
        return history
    return history[:2] + history[-(max_tail):]


def _read_log_tail(n=40):
    """Return last n lines of RUN_LOG."""
    try:
        lines = Path(RUN_LOG).read_text().splitlines()
        return '\n'.join(lines[-n:])
    except FileNotFoundError:
        return '(log not found)'


def parse_code_block(text):
    """
    Extract the # BEGIN MODIFIABLE ... # END MODIFIABLE block from text.
    Returns the block string (including markers) or None.
    """
    m = MARKER_RE.search(text)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Claude API interaction
# ---------------------------------------------------------------------------

def propose_change(client, history):
    """
    Ask Claude to propose a new get_hyperparams() block.
    Returns (description: str, new_block: str) or raises ValueError on bad output.
    """
    # Include current train_loop.py content as context on first real proposal
    current_code = TRAIN_SCRIPT.read_text()
    context_msg = (
        f"Current train_loop.py:\n```python\n{current_code}\n```\n\n"
        "Please propose the next experiment."
    )

    messages = history + [{"role": "user", "content": context_msg}]

    with client.messages.stream(
        model=MODEL_NAME,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        response = stream.get_final_message()

    # Extract text from response
    text = next(
        (b.text for b in response.content if b.type == "text"), ""
    )

    # Parse DESCRIPTION
    desc_match = re.search(r'DESCRIPTION:\s*(.+)', text)
    description = desc_match.group(1).strip() if desc_match else 'no description'

    # Parse CODE block
    new_block = parse_code_block(text)

    return description, new_block, text


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------

def research_loop(client, history):
    """Infinite loop: propose → apply → train → evaluate → record → repeat."""
    best_val_loss = float('inf')
    iteration = 0

    while True:
        iteration += 1
        print(f'\n{"="*60}')
        print(f'[loop] Iteration {iteration} | best_val_loss={best_val_loss:.6f}')
        print(f'{"="*60}')

        # --- 1. Propose change ---
        try:
            description, new_block, raw_response = propose_change(client, history)
        except anthropic.APIError as e:
            print(f'[api] Claude API error: {e}. Sleeping 30s ...')
            time.sleep(30)
            continue

        # --- Error: no code block ---
        if new_block is None:
            print('[error] Claude returned no code block.')
            history.append({"role": "user", "content": "Please propose the next experiment."})
            history.append({"role": "assistant", "content": raw_response})
            history.append({
                "role": "user",
                "content": (
                    "Your last response did not contain a valid "
                    "# BEGIN MODIFIABLE ... # END MODIFIABLE block. "
                    "Please respond with the required format:\n"
                    "DESCRIPTION: <one sentence>\n"
                    "CODE:\n# BEGIN MODIFIABLE\ndef get_hyperparams():\n    ...\n# END MODIFIABLE"
                )
            })
            history = prune_conversation_history(history)
            continue

        # --- Error: syntax check ---
        try:
            ast.parse(new_block)
        except SyntaxError as e:
            print(f'[error] Syntax error in proposed code: {e}')
            history.append({"role": "user", "content": "Please propose the next experiment."})
            history.append({"role": "assistant", "content": raw_response})
            history.append({
                "role": "user",
                "content": (
                    f"Your proposed code has a syntax error:\n{e}\n\n"
                    "Please fix it and resubmit."
                )
            })
            history = prune_conversation_history(history)
            continue

        # --- 2. Apply change ---
        try:
            apply_code_change(new_block)
        except ValueError as e:
            print(f'[error] Failed to apply code change: {e}')
            history.append({"role": "user", "content": "Please propose the next experiment."})
            history.append({"role": "assistant", "content": raw_response})
            history.append({
                "role": "user",
                "content": f"Failed to apply your code: {e}. Please try again."
            })
            history = prune_conversation_history(history)
            continue

        # --- 3. Commit ---
        commit_hash = git_commit(f'experiment: {description}')
        print(f'[git] Committed {commit_hash}: {description}')

        # --- 4. Train ---
        success, timed_out = run_training()

        # --- 5. Parse val_loss ---
        val_loss = parse_val_loss()

        # --- 6. Evaluate: keep or revert ---
        if val_loss is not None and val_loss < best_val_loss:
            status = 'keep'
            prev_best = best_val_loss
            best_val_loss = val_loss
            outcome_msg = (
                f"SUCCESS: val_loss={val_loss:.6f} (new best, improved from {prev_best:.6f}). "
                f"Change kept. Description: {description}"
            )
        else:
            status = 'revert'
            if timed_out:
                outcome_msg = (
                    f"TIMEOUT: Training exceeded {TRAINING_TIMEOUT}s. Change reverted. "
                    f"Description: {description}"
                )
            elif not success:
                log_tail = _read_log_tail()
                outcome_msg = (
                    f"CRASH: Training failed (non-zero exit). Change reverted. "
                    f"Description: {description}\n\nLog tail:\n{log_tail}"
                )
            else:
                val_str = f'{val_loss:.6f}' if val_loss is not None else 'N/A'
                outcome_msg = (
                    f"NO IMPROVEMENT: val_loss={val_str} "
                    f"(best={best_val_loss:.6f}). Change reverted. "
                    f"Description: {description}"
                )
            git_revert_one()

        # --- 7. Record ---
        record_result(commit_hash, val_loss, status, description)

        # --- 8. Update conversation history ---
        history.append({"role": "user", "content": "Please propose the next experiment."})
        history.append({"role": "assistant", "content": raw_response})
        history.append({"role": "user", "content": outcome_msg})

        # --- 9. Prune history ---
        history = prune_conversation_history(history)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Check prerequisites
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY environment variable is not set.')
        print('Please run: export ANTHROPIC_API_KEY=sk-ant-...')
        sys.exit(1)

    try:
        import anthropic as _anthropic_check  # noqa: F401
    except ImportError:
        print('ERROR: anthropic package not installed.')
        print('Please run: pip install anthropic')
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Setup git repo
    try:
        setup_git_repo()
    except subprocess.CalledProcessError as e:
        print(f'[git] Warning: git setup failed ({e}). Continuing without git push ...')

    # Apply baseline fixes and run baseline
    print('\n[setup] Applying baseline fixes ...')
    baseline_commit = apply_baseline_fixes()

    print('\n[setup] Running baseline training trial ...')
    success, timed_out = run_training()
    baseline_val_loss = parse_val_loss()

    if baseline_val_loss is None:
        log_tail = _read_log_tail()
        print(f'[setup] WARNING: Baseline training did not produce a val_loss.')
        print(f'[setup] Log tail:\n{log_tail}')
        baseline_val_loss_str = 'N/A'
    else:
        print(f'[setup] Baseline val_loss = {baseline_val_loss:.6f}')
        baseline_val_loss_str = f'{baseline_val_loss:.6f}'

    record_result(baseline_commit, baseline_val_loss, 'keep', 'baseline')

    # Prime conversation history with baseline context
    history = [
        {
            "role": "user",
            "content": (
                f"We are running an autonomous research loop to minimize val_loss for a "
                f"cosmological power spectrum neural network emulator. "
                f"The baseline hyperparameters achieved val_loss={baseline_val_loss_str}. "
                f"We will run 5-minute training trials. Your job is to propose changes to "
                f"get_hyperparams() one experiment at a time, and I will report the results back to you."
            )
        },
        {
            "role": "assistant",
            "content": (
                f"Understood! The baseline val_loss is {baseline_val_loss_str}. "
                f"I'll systematically explore hyperparameter changes to minimize it, "
                f"building on what works and avoiding what doesn't."
            )
        }
    ]

    print(f'\n[loop] Starting research loop. Press Ctrl-C to stop.')
    print(f'[loop] Results will be saved to: {RESULTS_TSV}')
    print(f'[loop] Training logs: {RUN_LOG}')

    try:
        research_loop(client, history)
    except KeyboardInterrupt:
        print('\n[loop] Interrupted by user. Exiting.')
        print(f'[loop] Final results in: {RESULTS_TSV}')


if __name__ == '__main__':
    main()
