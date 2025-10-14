# From https://github.com/Andrewwango/ssibench/blob/44af995f4ce3e09bfbc346c1646f8dc71dde9618/README.md

import deepinv as dinv
import torch

if __name__ == "__main__":
    dataset = dinv.datasets.FastMRISliceDataset(
        "FastMRI/knee/singlecoil_train",
        slice_index="middle",
        rng=torch.Generator().manual_seed(0),
    )

    dataset.save_simple_dataset("FastMRI-Slices/fastmri_knee_singlecoil.pt")
