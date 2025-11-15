from pathlib import Path
import torch
import deepinv as dinv
from torchvision import datasets, transforms
import re


def get_measurement_dir(**config):
    dataset_filename = f"acceleration={config['acceleration']}_noise={config['noise']}" if "MRI" in config[
        'operation'] else \
        f"oversampling_ratio={config['oversampling_ratio']}_n_images={config['n_images']}_"
    measurement_dir = (
            Path(".")
            / "ckpts"
            / config["operation"]
            / (re.sub(r'\b_data_augment\b', '', config["dataset_name"]) + "_" + dataset_filename)
    ) # (re.sub(r'\b_data_augment\b', '', config["dataset_name"]) remplace _data_augment par rien ("") dans config["dataset_name"]).
    return measurement_dir

def get_Lphysics(device, rng=None, **config):
    measurement_dir = get_measurement_dir(**config)
    if "CS" in config["operation"]:
        physics = dinv.physics.CompressedSensing(
            m=int(config["oversampling_ratio"] * 784),
            fast=False,
            channelwise=True,
            img_size=(1, 28, 28),
            noise_model=dinv.physics.noise.GaussianNoise(sigma=config["noise"]) if config[
                "noise"] else dinv.physics.noise.ZeroNoise(),
            device=device,
            rng=rng,
        )
        try:
            physics.load_state_dict(
            torch.load(measurement_dir / f"physics0.pt",
                       weights_only=True,
                       map_location=device,
                       )
            )
        except:
            pass
    elif "inpainting" in config["operation"]:
        mask = torch.load(measurement_dir / f"physics0.pt",
                          weights_only=True,
                          map_location=device,
                          )["mask"] if "DIV2K" in config["dataset_name"] else config["oversampling_ratio"]
        physics = dinv.physics.Inpainting((1, 28, 28) if "MNIST" in config["dataset_name"] else (3, 128, 128),
                                          mask=mask, device=device,
                                          noise_model=dinv.physics.noise.GaussianNoise(sigma=config["noise"]) if config[
                                              "noise"] else dinv.physics.noise.ZeroNoise(),
                                          rng=rng
                                          )
    else:
        raise ValueError("Unknown operation")

    return physics

def get_LDataset(device, physics=None, **config):
    dataset_filename = f"oversampling_ratio={config['oversampling_ratio']}_n_images={config['n_images']}_"
    measurement_dir = (
            Path(".")
            / "ckpts"
            / config["operation"]
            / (re.sub(r'_data_augment', '', config["dataset_name"]) + "_" + dataset_filename)
    )
    print(f"Loading dataset from {measurement_dir}")

    list_transf = [transforms.ToTensor()]
    if "shift_invariant" in config["dataset_name"]:
        list_transf.append(
            transforms.Lambda(
                lambda x: torch.roll(
                    x,
                    (torch.randint(0, 28, (1,)).item(), torch.randint(0, 28, (1,)).item()),
                    dims=(-2, -1),
                )
            )
        )
    elif "data_augment" in config["dataset_name"]:
        list_transf += [transforms.Resize(256), transforms.RandomCrop(128)]

    transform = transforms.Compose(list_transf)
    if "MNIST" in config["dataset_name"]:
        if "Fashion" in config["dataset_name"]:
            train_dataset = datasets.FashionMNIST(root="", train=True, transform=transform, download=False)
            test_dataset = datasets.FashionMNIST(root="", train=False, transform=transform, download=False)
        else:
            train_dataset = datasets.MNIST(root="", train=True, transform=transform, download=False)
            test_dataset = datasets.MNIST(root="", train=False, transform=transform, download=False)
            if "shift_invariant" in config["dataset_name"]:
                train_dataset = datasets.MNIST(root="", train=True, transform=transform, download=False)
                test_dataset = datasets.MNIST(root="", train=False, transform=transform, download=False)
                train_dataset = torch.utils.data.Subset(train_dataset, range(config["n_images"]))
                test_dataset = torch.utils.data.Subset(test_dataset, range(100))
            else:
                dataset_path = dinv.datasets.generate_dataset(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    physics=physics,
                    save_physics_generator_params=True,
                    overwrite_existing=False,
                    train_datapoints=config["n_images"],
                    test_datapoints=100,
                    device=device,
                    save_dir=measurement_dir,
                    batch_size=500,
                )
                train_dataset = dinv.datasets.HDF5Dataset(path=dataset_path, train=True)
                test_dataset = dinv.datasets.HDF5Dataset(path=dataset_path, train=False)

    elif "DIV2K" in config["dataset_name"]:
        if "data_augment" in config["dataset_name"]:
            train_dataset = dinv.datasets.DIV2K(root="DIV2K", mode="train", transform=transform, download=False)
            test_dataset = dinv.datasets.DIV2K(root="DIV2K", mode="val", transform=transform, download=False)
        else:
            train_dataset = dinv.datasets.HDF5Dataset(path=measurement_dir / "dataset0.h5", train=True)
            test_dataset = dinv.datasets.HDF5Dataset(path=measurement_dir / "dataset0.h5", train=False)

    return train_dataset, test_dataset

def get_Lmodel(device, **config):
    if "MNIST" in config["dataset_name"]:
        C = 1
    elif "DIV2K" in config["dataset_name"]:
        C = 3
    else:
        raise ValueError("Unknown dataset")

    backbone = (
        dinv.models.UNet_equi(
            in_channels=C,
            out_channels=C,
            residual="no_residual" not in config["model"],
            bias=True,
            batch_norm="no_norm" not in config["model"],
            scales=4,
            circular_padding="zeros_padding" not in config["model"],
            device=device,
            equivariance=True  # attention il faut mettre true mais ca change en layer norm
        ).to(device)
        if "equi" in config["model"]
        else dinv.models.UNet(
            in_channels=C,
            out_channels=C,
            residual="no_residual" not in config["model"],
            bias=True,
            batch_norm="no_norm" not in config["model"],
            layer_norm=True,
            scales=4,
            circular_padding=False,
        ).to(device)
    )
    model = dinv.models.MoDL(denoiser=backbone, num_iter=3).to(device) if "unroll" in config["model"] else dinv.models.ArtifactRemoval(backbone, pinv=False)

    return model

def get_Llosses(device, **config):
    losses = []
    for loss in config["losses_name"]:
        if "mse" in loss:
            losses.append(dinv.loss.SupLoss(**config["losses_params"]["mse"]))
        elif "mc" in loss:
            losses.append(dinv.loss.MCLoss(**config["losses_params"]["mc"]))
        elif "Sure" in loss:
            losses.append(dinv.loss.SureGaussianLoss(**config["losses_params"]["Sure"]))
        elif "ei" in loss:
            losses.append(dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=1), **config["losses_params"]["ei"]))
        elif "Split" in loss:
            mask_generator = dinv.physics.generator.DeterministSplittingMaskGenerator(
                device=device, tensor_size=(1, int(config["oversampling_ratio"] * 28 * 28)), **config["losses_params"]["mask_generator"]
            ) if "CS" in config["operation"] else None
            losses.append(
                dinv.loss.SplittingLoss(
                    mask_generator=mask_generator, **config["losses_params"]["Split"]
                )
            )
    return losses
