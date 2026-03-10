import numpy as np
from datetime import timedelta
from pathlib import Path
from sunbird.emulators import FCN
from sunbird.emulators.train import FCNTrainer
from sunbird.data import ArrayDataModule
from sunbird.data.transforms_array import LogTransform, ArcsinhTransform
import torch
import argparse

torch.set_float32_matmul_precision('high')

def _build_transform(transform_name):
    if transform_name is None:
        return None
    if transform_name == 'log':
        return LogTransform()
    if transform_name == 'arcsinh':
        return ArcsinhTransform()
    raise ValueError(f'Unknown transform: {transform_name}')


def load_data(observable_name):
    data_dir = Path('/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.3/abacus/compressed')
    data = np.load(data_dir / f'{observable_name}.npy', allow_pickle=True).item()
    x = data['x']['data'] # shape (ncosmo, nhod, nparams)
    y = data['y']['data'] # shape (ncosmo, nhod, npoles, nk)

    # select first 6 cosmologies for testing, rest for training/validation
    x_train = x[6:, :]
    y_train = y[6:, :]

    # now reshape to (nsamples, n_features)
    x_train = x_train.reshape(-1, x_train.shape[-1])
    y_train = y_train.reshape(-1, y_train.shape[-2] * y_train.shape[-1])

    print('Loaded train x, y with shapes:', x_train.shape, y_train.shape)

    covariance_y = data['covariance_y']['data']
    covariance_y = covariance_y.reshape(covariance_y.shape[0], -1)
    covariance_matrix = np.cov(covariance_y, rowvar=False)
    print('Loaded covariance matrix with shape:', covariance_matrix.shape)

    return x_train, y_train, covariance_matrix


# BEGIN MODIFIABLE
def get_hyperparams():
    """Tunable hyperparameters. Valid options documented in docstring.

    act_fn options: learned_sigmoid, SiLU, GELU, ELU, Mish, Tanh, ReLU, LeakyReLU, SELU, Hardswish
    loss options: weighted_mae, weighted_mse, GaussianNLoglike, mse, rmse, mae
    transform_input/output: None or 'arcsinh' (log is INVALID: inputs/multipoles can be negative)
    """
    return dict(
        learning_rate=1e-3,
        n_hidden=[512, 512, 512, 512],
        dropout_rate=0.0,
        weight_decay=0,
        batch_size=128,
        val_fraction=0.1,
        act_fn='learned_sigmoid',        # or: SiLU, GELU, ELU, Mish, ReLU, etc.
        loss='weighted_mae',             # or: weighted_mse, GaussianNLoglike, mse, mae
        transform_input=None,            # or: 'arcsinh'
        transform_output='arcsinh',      # arcsinh compresses dynamic range, handles negatives safely
        scheduler_patience=10,
        scheduler_factor=0.5,
        scheduler_threshold=1e-6,
        gradient_clip_val=0.5,
    )
# END MODIFIABLE


def TrainFCN(x, y, covariance_matrix, learning_rate, n_hidden, dropout_rate,
             weight_decay, batch_size=128, val_fraction=0.1, act_fn='learned_sigmoid',
             loss='weighted_mae', transform_input=None, transform_output=None,
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

    dump_dir = '/pscratch/sd/e/epaillas/dump/train'
    trainer = FCNTrainer(
        checkpoint_dir=dump_dir,
        max_epochs=5000,
        devices=1,
        max_time=timedelta(minutes=max_time_minutes),
        logger='tensorboard',
        log_dir=dump_dir,
        gradient_clip_val=gradient_clip_val,
    )
    val_loss = trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    return val_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train FCN for EMC observables.')
    parser.add_argument('-s', '--statistic', type=str, default='spectrum', help='Statistic to train on.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--max_time_minutes', type=float, default=5, help='Maximum training time in minutes. Defaults to 5.')
    args = parser.parse_args()

    x, y, covariance_matrix = load_data(args.statistic)

    hp = get_hyperparams()
    val_loss = TrainFCN(x, y, covariance_matrix, seed=args.seed,
                        max_time_minutes=args.max_time_minutes, **hp)
    print(f"val_loss: {val_loss:.6f}")
