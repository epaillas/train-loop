# trainloop — Autonomous Hyperparameter Research for Power Spectrum Emulator

An autonomous research loop that uses the Claude API to iteratively tune hyperparameters of a fully-connected neural network (FCN) emulator for cosmological power spectrum multipoles.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md).

---

## Overview

`researcher.py` runs an infinite loop:
1. Calls Claude (`claude-opus-4-6`) to propose a change to `get_hyperparams()` in `train_loop.py`
2. Commits the change to git
3. Runs a 5-minute training trial
4. If `val_loss` improves → keeps the commit; otherwise → reverts it
5. Reports the outcome back to Claude and repeats

The baseline `val_loss` is ~0.189. All experiment results are logged to `results.tsv`.

---

## Files

| File | Description |
|---|---|
| `train_loop.py` | Trains an FCN on power spectrum data; hyperparameters live in `get_hyperparams()` |
| `researcher.py` | Autonomous research loop (calls Claude API, manages git, runs training) |
| `results.tsv` | Experiment log: commit, val_loss, status (keep/revert), description |
| `run.log` | stdout/stderr of the most recent training run |

---

## Usage

```bash
# Install dependency
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run the loop (indefinitely; Ctrl-C to stop)
cd /global/cfs/cdirs/desicollab/users/epaillas/code/trainloop
python researcher.py
```

Monitor progress:
```bash
tail -f run.log       # live training output
cat results.tsv       # experiment history
git log --oneline     # commit history
```

---

## How It Works

### `train_loop.py`

All tunable hyperparameters are isolated in `get_hyperparams()` between marker comments:

```python
# BEGIN MODIFIABLE
def get_hyperparams():
    return dict(
        learning_rate=1e-3,
        n_hidden=[512, 512, 512, 512],
        act_fn='learned_sigmoid',
        loss='weighted_mae',
        # ... etc.
    )
# END MODIFIABLE
```

Claude only ever rewrites the body of this function. Everything else (`load_data`, `TrainFCN`, imports) is off-limits.

### Valid Hyperparameter Values

| Parameter | Valid values |
|---|---|
| `act_fn` | `learned_sigmoid`, `SiLU`, `GELU`, `ELU`, `Mish`, `Tanh`, `ReLU`, `LeakyReLU`, `SELU`, `Hardswish` |
| `loss` | `weighted_mae`, `weighted_mse`, `GaussianNLoglike`, `mse`, `rmse`, `mae` |
| `transform_input` / `transform_output` | `None` or `'arcsinh'` (**not** `'log'` — multipoles can be negative at large k) |

### `results.tsv` Format

```
commit    val_loss    status    description
a1b2c3d   0.189140    keep      baseline
b2c3d4e   0.184200    keep      lower learning rate 3e-4 for finer convergence
c3d4e5f   0.191000    revert    wider architecture [1024,1024,512,512] too slow
```

---

## Prerequisites

- Python 3.10+
- `pip install anthropic`
- `ANTHROPIC_API_KEY` environment variable
- Access to training data at `/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/`
- GPU node with checkpoint dir at `/pscratch/sd/e/epaillas/dump/train`
