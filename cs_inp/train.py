import torch
import deepinv as dinv
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import os
import re
from get_linear_physics import get_Lphysics, get_LDataset, get_Lmodel, get_Llosses


def train(device, **config):
    print("Start training")
    # %%
    # Set up the directories
    dataset_filename = f"acceleration={config['acceleration']}_noise={config['noise']}" if "MRI" in config['operation'] else \
        f"oversampling_ratio={config['oversampling_ratio']}_n_images={ config['n_images']}_"
    measurement_dir = (
        Path(".")
        / "ckpts"
        / config["operation"]
        / (re.sub(r'\b_data_augment\b', '', config["dataset_name"]) + "_" + dataset_filename)
    )  #(re.sub(r'\b_data_augment\b', '', config["dataset_name"]) remplace _data_augment par rien ("") dans config["dataset_name"]).

    # %%
    # Set up the data with the physics
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    physics = get_Lphysics(device, rng, **config)

    train_dataset, test_dataset = get_LDataset(device, physics, **config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=4 if torch.cuda.is_available() and (__name__ == "__main__") else 0,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=10,
        num_workers=4 if torch.cuda.is_available() and (__name__ == "__main__") else 0,
        shuffle=False,
    )

    # %%
    # Set up the denoiser network
    # ---------------------------------------------------------------
    #
    model = get_Lmodel(device, **config)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {params} trainable parameters")
    # %%
    # choose training loss
    losses = get_Llosses(device, **config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(config["epochs"] * 0.85)), gamma=0.1)

    # %%
    # Train the network
    # --------------------------------------------
    #
    verbose = True  # print training information
    wandb_vis = "STY" in os.environ  # plot curves and images in Weight&Bias, "STY" in os.environ determines whether we are inside a screen session or not.

    if wandb_vis:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="ICLR",  # config["operation"], #
            name="_".join(config["losses_name"]),
            group=config["dataset_name"]
            + "_"
            + config["operation"]
            + "_"
            + "n_images="
            + str(config["n_images"]),
            # track hyperparameters and run metadata
            config=config,
        )

    metrics = [dinv.loss.PSNR()]

    # Initialize the trainer
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        epochs=config["epochs"],
        scheduler=scheduler,
        online_measurements=("data_augment" in config["dataset_name"]) or ("shift_invariant" in config["dataset_name"]),
        losses=losses,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        eval_interval=1,
        display_losses_eval="mse" in config["losses_name"],
        plot_images=False,
        save_path=str(measurement_dir / config["method"] / (config["model"] + "_" + "_".join(config["losses_name"]))), # + (str(config["losses_params"]["mask_generator"]["split_ratio"]) if "Split" in config["losses_name"] else ""))),
        ckp_interval=1,
        verbose=verbose,
        show_progress_bar=__name__ == "__main__",
        wandb_vis=wandb_vis,
        check_grad=True,
    )

    # Train the network
    model = trainer.train()

    save_path = trainer.save_path
    config["date"] = Path(save_path).name  # save the date of the training

    best_model_ckpt = save_path + "/best_model.pth.tar"

    checkpoint= torch.load(
        best_model_ckpt, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    best_model_perf = dinv.test(
        model=model,
        test_dataloader=test_dataloader,
        physics=physics,
        metrics=None,
        online_measurements=False,
        device=device,
        plot_images=False,
        verbose=True,
        show_progress_bar=False,
        compare_no_learning=True,
        no_learning_method="A_adjoint")

    config["eval_metric"] = best_model_perf

    return trainer