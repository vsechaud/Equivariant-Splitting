from losses import SplitR2RLoss
from training import Trainer
from metrics import EQUIV
from models import EquivariantReconstructor, UNet

import deepinv as dinv
from deepinv.models import UNet as UNetDI
import wandb
import mlflow
import torch
import torch.utils.data
from torchvision.transforms import InterpolationMode
import pandas as pd

import os
import sys
from pathlib import Path
import yaml


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <config>")
        sys.exit(1)

    config_name = sys.argv[1]

    with open(f"configs/{config_name}.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Optional checkpoint for resuming training or post-training evaluation
    ckpt = config.get("checkpoint_path", None)

    # Define training parameters
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    rng = torch.Generator(device=device).manual_seed(0)
    rng_cpu = torch.Generator(device="cpu").manual_seed(0)
    acceleration = config["acceleration"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    loss_name = config["loss"]
    img_size = (320, 320)

    # Define MRI physics $A$ and mask generator $M$ according to scenario
    center_fraction = config["center_fraction"]
    physics_generator = dinv.physics.generator.GaussianMaskGenerator(
        img_size=img_size,
        acceleration=acceleration,
        center_fraction=center_fraction,
        rng=rng,
        device=device,
    )
    physics = dinv.physics.MRI(img_size=img_size, device=device)

    sigma = config["noise_sigma"]
    physics.update(**physics_generator.step())
    physics.noise_model = dinv.physics.GaussianNoise(sigma, rng=rng)

    # Define model $f_\theta$
    denoiser_kind = config["denoiser_kind"]
    denoiser_in_channels = 2
    denoiser_out_channels = 2
    denoiser_bias = True
    denoiser_residual = True
    denoiser_normalization = config["denoiser_normalization"]
    unet_scales = 4
    if denoiser_kind in ["UNet", "UNet-R2"]:
        if denoiser_normalization == "layer_norm_af":
            batch_norm = True
            norm_type = "layer_norm_af"
        elif denoiser_normalization == "batch_norm":
            batch_norm = True
            norm_type = "batch_norm"
        elif denoiser_normalization == "none":
            batch_norm = False
            norm_type = None
        else:
            raise ValueError(f"Invalid normalization {denoiser_normalization}.")
        if denoiser_kind == "UNet":
            denoiser = UNet(
                in_channels=denoiser_in_channels,
                out_channels=denoiser_out_channels,
                scales=unet_scales,
                bias=denoiser_bias,
                residual=denoiser_residual,
                batch_norm=batch_norm,
                norm_type=norm_type,
            )
        else:
            raise ValueError(f"Invalid denoiser kind {denoiser_kind}.")
    elif denoiser_kind == "UNet-DeepInverse":
        if denoiser_normalization == "batch_norm":
            batch_norm = True
        elif denoiser_normalization == "none":
            batch_norm = False
        else:
            raise ValueError(f"Invalid normalization {denoiser_normalization}.")
        denoiser = UNetDI(
            in_channels=denoiser_in_channels,
            out_channels=denoiser_out_channels,
            scales=unet_scales,
            bias=denoiser_bias,
            residual=denoiser_residual,
            batch_norm=batch_norm,
        )
    elif denoiser_kind == "IdentityDenoiser":

        class IdentityDenoiser(dinv.models.Denoiser):
            def forward(self, x, sigma=None, **kwargs):
                return x

        denoiser = IdentityDenoiser()
    else:
        raise ValueError(f"Invalid denoiser kind {denoiser_kind}.")

    denoiser_reynolds_group = config.get("denoiser_reynolds_group", "Trivial")
    if denoiser_reynolds_group != "Trivial":
        if denoiser_reynolds_group != "D4":
            raise ValueError(f"Invalid group {denoiser_reynolds_group}.")
        denoiser = dinv.models.EquivariantDenoiser(denoiser, random=True)

    reconstructor_kind = config["reconstructor_kind"]
    if reconstructor_kind == "MoDL":
        model = dinv.models.MoDL(denoiser=denoiser, num_iter=3)
    elif reconstructor_kind == "ArtifactRemoval":
        model = dinv.models.ArtifactRemoval(
            backbone_net=denoiser, mode="adjoint", device=device
        )
    else:
        raise ValueError(f"Invalid reconstructor kind {reconstructor_kind}.")

    reconstructor_reynolds_group = config.get("reconstructor_reynolds_group", "Trivial")
    if reconstructor_reynolds_group != "Trivial":
        kwargs = {}
        if reconstructor_reynolds_group == "C360":
            kwargs["transform"] = dinv.transform.Rotate(
                multiples=1.0, interpolation_mode=InterpolationMode.BILINEAR
            )
        elif reconstructor_reynolds_group != "D4":
            raise ValueError(f"Invalid group {reconstructor_reynolds_group}.")

        eval_mode = config.get("reconstructor_reynolds_eval_mode", "same")
        model = EquivariantReconstructor(
            model, random=True, eval_mode=eval_mode, **kwargs
        )

    model.to(device)

    # Define dataset
    dataset = dinv.datasets.SimpleFastMRISliceDataset(
        "./FastMRI-Slices", file_name=f"fastmri_knee_singlecoil.pt"
    )
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (900, 73), generator=rng_cpu
    )

    # Simulate and save random measurements
    save_dir = "/lustre/fswork/projects/rech/zqk/uqv91qh/wd/data"
    dataset_filename = (
        f"dataset_knee_single-noisy_acceleration={acceleration}_sigma={sigma}"
    )
    dataset_path = dinv.datasets.generate_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        physics=physics,
        physics_generator=None,
        save_physics_generator_params=True,
        overwrite_existing=False,
        device=device,
        save_dir=save_dir,
        batch_size=1,
        dataset_filename=dataset_filename,
    )
    print(dataset_path, device)
    train_dataset = dinv.datasets.HDF5Dataset(
        dataset_path, split="train", load_physics_generator_params=True
    )
    test_dataset = dinv.datasets.HDF5Dataset(
        dataset_path, split="test", load_physics_generator_params=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=rng_cpu
    )
    test_dataloader_rng = torch.Generator().manual_seed(0)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False
    )

    match loss_name:
        case "sup":
            loss = [dinv.loss.SupLoss()]

        case "sure":
            loss = [dinv.loss.SureGaussianLoss(sigma=sigma, tau=config["sure_tau"])]

        case "r2r-ssdu":
            split_generator = dinv.physics.generator.GaussianMaskGenerator(
                img_size=img_size,
                acceleration=2,
                center_fraction=0.0,
                rng=rng,
                device=device,
            )
            mask_generator = (
                dinv.physics.generator.MultiplicativeSplittingMaskGenerator(
                    (1, *img_size), split_generator, device=device
                )
            )
            eval_n_samples = config.get("splitting_eval_n_samples", 10)
            loss = [
                dinv.loss.SplittingLoss(),
                SplitR2RLoss(
                    mask_generator=mask_generator,
                    noise_model=dinv.physics.GaussianNoise(sigma),
                    alpha=0.2,
                    weight=1.0,
                    eval_n_samples=eval_n_samples,
                ),
            ]
            model = loss[-1].adapt_model(model)

        case "robust-ei":
            n_trans = config["ei_n_trans"]
            interpolation_mode = config["ei_interpolation_mode"]
            interpolation_mode = interpolation_mode.upper()
            interpolation_mode = getattr(InterpolationMode, interpolation_mode)
            ei_group = config["ei_group"]
            if ei_group == "C360":
                multiples = 1.0
                with_flips = False
            elif ei_group == "D4":
                multiples = 90.0
                with_flips = True
            else:
                raise ValueError(f"Invalid group {ei_group}.")
            transform = dinv.transform.Rotate(
                limits=360.0,
                n_trans=n_trans,
                interpolation_mode=interpolation_mode,
                multiples=multiples,
            )
            if with_flips:
                if n_trans != 1:
                    raise ValueError(f"Invalid n_trans {n_trans}.")
                transform = transform * dinv.transform.Reflect(
                    n_trans=n_trans, dims=[-1]
                )
            loss = [
                dinv.loss.SureGaussianLoss(sigma=sigma, tau=config["sure_tau"]),
                dinv.loss.EILoss(transform=transform),
            ]

        case _:
            raise ValueError(f"Invalid loss {loss_name}.")

    # Define metrics
    metrics = [dinv.metric.PSNR(complex_abs=True), dinv.metric.SSIM(complex_abs=True)]

    test_metrics = [
        EQUIV(
            transform=dinv.transform.Rotate(n_trans=1, multiples=90, positive=True)
            * dinv.transform.Reflect(n_trans=1, dim=[-1]),
            metric=dinv.metric.MSE(),
            n_samples=8,
            db=True,
        )
    ]

    mlops_run_name = config_name
    mlops_config = {
        **config,
    }

    wandb_vis = False
    if acceleration == 8:
        mlops_scope_name = "MRIx8"
    elif acceleration == 6:
        mlops_scope_name = "MRIx6"
    else:
        raise ValueError(f"Invalid acceleration {acceleration}.")

    if wandb_vis:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=mlops_scope_name,
            name=mlops_run_name,
            config=mlops_config,
        )

    mlflow_vis = True
    if mlflow_vis:
        mlflow.set_experiment(mlops_scope_name)

        # Start a new MLflow run to track this script
        mlflow.start_run(run_name=mlops_run_name)

        # Log hyperparameters (skip Nones)
        mlflow.log_params(mlops_config)

    if list(model.parameters()) != []:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config["weight_decay"],
        )
    else:
        optimizer = None
    scheduler_name = config["scheduler"]
    if scheduler_name == "80%/2":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(epochs * 0.8), gamma=0.5
        )
    elif scheduler_name == "cosine_annealing":
        eta_min = config["scheduler_final_lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
    elif scheduler_name == "exponential":
        final_lr = config["scheduler_final_lr"]
        gamma = (final_lr / lr) ** (1 / epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Invalid scheduler {scheduler_name}.")

    # Define the trainer
    display_losses_eval = "r2r-ssdu" not in loss_name  # Not supported yet
    trainer = Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        epochs=epochs,
        losses=loss,
        metrics=metrics,
        device=device,
        ckpt_pretrained=ckpt,
        wandb_vis=wandb_vis,
        mlflow_vis=mlflow_vis,
        save_path="./results",
        display_losses_eval=display_losses_eval,
        merge_losses=False,
        ckp_interval=50,
    )

    _ = trainer.train()
    if epochs != 0:
        _ = trainer.load_best_model()

    test_save_path = f"{trainer.save_path}/test"
    os.makedirs(test_save_path, exist_ok=True)
    model.eval()
    trainer.metrics += test_metrics
    metrics = trainer.test(
        test_dataloader, save_path=test_save_path, compare_no_learning=True
    )

    metrics["config_name"] = config_name

    mlflow.log_dict(metrics, "metrics_dict.json")

    mlflow.log_text(str(metrics), "metrics.txt")

    # metrics is a dict that can be understood as a row of a dataframe
    df = pd.DataFrame([metrics])
    mlflow.log_table(data=df, artifact_file="metrics_tbl.json")
