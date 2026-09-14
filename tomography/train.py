import os
import random
import sys
from pathlib import Path
import torch
import deepinv as dinv
import json
import pandas as pd

import yaml

from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import InterpolationMode

try:
    import mlflow
except ImportError:
    mlflow = None

from physics import (
    get_physics,
    NOISE_SIGMA,
)
from r2rsplitting import SplitR2RLoss
from splitting_masks import MaskTomography
from metrics import EQUIV
from training import Trainer
from models import UNet, EquivariantReconstructor


def convert_config_to_strings(d):
    if isinstance(d, dict):
        return {k: convert_config_to_strings(v) for k, v in d.items()}
    elif callable(d):
        return d.__class__.__name__
    return d


# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Load the config
if len(sys.argv) == 2:
    config_name = sys.argv[1]
else:
    raise ValueError("Unexpected number of arguments. Usage: python train.py <config_name>")

config_path = f"./configs/{config_name}.yaml"
with open(config_path, "r") as stream:
    config = yaml.safe_load(stream)

physics = get_physics(device)

trainval_dataset = dinv.datasets.HDF5Dataset("./LIDC_IDRI-Tomography/dinv_dataset0.h5", train=True)

# NOTE: It is meant to be a deterministic function.
# Reproducibility is guaranteed as long as trainval_dataset is always the same.
def get_splits(trainval_dataset, train_split_size, val_split_size, seed=0):
    generator = torch.Generator().manual_seed(seed)
    train_dataset, eval_dataset = torch.utils.data.random_split(
        trainval_dataset, [train_split_size, val_split_size], generator=generator
    )
    return train_dataset, eval_dataset


train_dataset, eval_dataset = get_splits(
    trainval_dataset=trainval_dataset,
    train_split_size=800,
    val_split_size=100,
    seed=1,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=5,
    num_workers=1,
    shuffle=True,
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=5,
    num_workers=1,
    shuffle=False,
)

# Set up the denoiser network
# ---------------------------------------------------------------
#

denoiser = UNet(
    in_channels=1,
    out_channels=1,
    scales=4,
    bias=True,
    residual=True,
    batch_norm=False,
)

# Use the pseudo-inverse instead of the adjoint for the initialization of MoDL
def pseudo_inverse_init(y, physics):
    x_hat = physics.A_dagger(y)
    return { "est": (x_hat, x_hat) }

model = dinv.models.MoDL(denoiser=denoiser, num_iter=3)
model.custom_init = pseudo_inverse_init

if config["equivariance"]:
    model = EquivariantReconstructor(model, random=True, eval_mode="same")

model.to(device)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {params} trainable parameters")

# Get the loss
loss_name = config["loss"]
match loss_name:
    case "Supervised":
        losses = [dinv.loss.SupLoss()]
    case "ES":
        # make the ordering fixed
        mask_generator = MaskTomography((1, 363, 50), split_ratio=0.6, device=device)

        # Assume that a single splitting loss is present in the losses
        losses = [
            dinv.loss.SplittingLoss(),
            SplitR2RLoss(
                mask_generator=mask_generator,
                noise_model=dinv.physics.GaussianNoise(NOISE_SIGMA),
                alpha=0.2,
                weight=1.0,
                eval_n_samples=10,
            )
        ]
        model = losses[-1].adapt_model(model)
    case "EI":
        transform = dinv.transform.Rotate(
            limits=360.0,
            n_trans=1,
            interpolation_mode=InterpolationMode.BILINEAR,
            multiples=1.0,
        )
        losses = [
            # 10% of the noise standard deviation 0.001
            dinv.loss.SureGaussianLoss(sigma=NOISE_SIGMA, tau=0.0001),
            dinv.loss.EILoss(transform=transform),
        ]
    case _:
        raise ValueError(f"Loss {loss_name} not supported")

# choose optimizer and scheduler
if list(model.parameters()) != []:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-8,
    )
else:
    optimizer = None
scheduler = None
# Train the network
# --------------------------------------------
#

# Configuration for Logging (MLflow), disabled if mlflow is not installed
use_mlflow = mlflow is not None
if use_mlflow:
    mlflow.set_experiment("Equivariant-Splitting_CT")

    # Start a new MLflow run to track this script
    mlflow.start_run(run_name=config_name)

    # Log hyperparameters (convert dict to strings to avoid serialization issues)
    mlflow.log_params(convert_config_to_strings(config))
else:
    print("mlflow is not installed, logging to MLflow is disabled")


metrics = [
    dinv.metric.PSNR(max_pixel=1.0),
    dinv.metric.SSIM(),
    dinv.loss.MCLoss(dinv.metric.MSE()),
]

test_metrics = [
    EQUIV(
        transform=dinv.transform.Rotate(
            n_trans=1, multiples=90, positive=True
        ) * dinv.transform.Reflect(n_trans=1, dim=[-1]),
        metric=dinv.metric.MSE(),
        n_samples=8,
        db=True,
    )
]

epochs = config["epochs"]
trainer = Trainer(
    model=model,
    physics=physics,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    epochs=epochs,
    losses=losses,
    metrics=metrics,
    device=device,
    online_measurements=False,
    ckpt_pretrained=None,
    wandb_vis=False,
    mlflow_vis=use_mlflow,
    save_path=f"./results/{config_name}",
    display_losses_eval=loss_name != "ES",
    merge_losses=False,
    ckp_interval=50,
    no_learning_method="A_dagger",
)

# Train the network
_ = trainer.train()
if epochs != 0:
    _ = trainer.load_best_model()

save_path = Path(trainer.save_path)
config["date"] = save_path.name  # save the date of the training

config2save = convert_config_to_strings(config)
with open(save_path / "config.json", "w") as f:
    json.dump(config2save, f)

trainer.plot_images = True

# Capture test metrics for logging
test_save_path = save_path / "test"
os.makedirs(test_save_path, exist_ok=True)
model.eval()
trainer.metrics += test_metrics
test_metrics = trainer.test(eval_dataloader, save_path=test_save_path, compare_no_learning=True)

# Log final metrics to MLflow
test_metrics["config_name"] = config_name

if use_mlflow:
    # Log as JSON artifact
    mlflow.log_dict(test_metrics, "metrics_dict.json")

    # Log as Text artifact
    mlflow.log_text(str(test_metrics), "metrics.txt")

    # Log as Table
    # metrics is a dict that can be understood as a row of a dataframe
    df = pd.DataFrame([test_metrics])
    mlflow.log_table(data=df, artifact_file="metrics_tbl.json")
