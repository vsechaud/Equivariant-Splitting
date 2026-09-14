import torch
import torch.utils.data
import torchvision.transforms.functional as TF
import numpy
import deepinv as dinv

import bisect
import random

from lidc_idri import LidcIdriSliceDataset
from physics import get_physics

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)

    def transform(x: numpy.ndarray) -> torch.Tensor:
        # numpy.ndarray -> torch.Tensor
        x = torch.from_numpy(x)

        # (512, 512) -> (1, 512, 512)
        x = x.unsqueeze(0)

        # (1, 512, 512) -> (1, 256, 256)
        x = TF.resize(
            x, [256, 256], antialias=True, interpolation=TF.InterpolationMode.BICUBIC
        )

        # int16 -> float32
        x = x.float()

        # Normalize values to [0, 1].
        # Clip values below -1000 (air) and above 1000 (bone)
        x = (1000 + torch.clamp(x, min=-1000, max=1000)) / 2000

        return x

    dataset: LidcIdriSliceDataset = LidcIdriSliceDataset(
        root="./LIDC_IDRI", transform=transform, hu=True
    )

    # Select one slice per patient (arbitrarily but deterministically).
    # For each patient, the selected slice is the smallest one for the default ordering on LidcIdriSliceDataset.SliceSampleIdentifier.
    # If this is not possible, i.e. if two different minimal slices are present for the same patient, we raise an error as there is no deterministic way to select one.
    indices: list[int] = []
    minima: dict[str, tuple[LidcIdriSliceDataset.SliceSampleIdentifier, bool, int]] = {}
    idx: int
    ordinal: LidcIdriSliceDataset.SliceSampleIdentifier
    for idx, ordinal in enumerate(dataset.sample_identifiers):
        patient_id = ordinal.patient_id
        if patient_id not in minima:
            minima[patient_id] = (ordinal, False, idx)
        else:
            minimum_ordinal, _, _ = minima[patient_id]
            if ordinal < minimum_ordinal:
                minima[patient_id] = (ordinal, False, idx)
            elif ordinal == minimum_ordinal:
                # Duplicates are allowed as long as they are not the minimum. We track their presence here.
                # NOTE: There is probably no duplicate anyway.
                minima[patient_id] = (ordinal, True, idx)
            else:
                assert (
                        ordinal > minimum_ordinal
                ), "Metadata are expected to be comparable."

    is_duplicate: bool
    idx: int
    # NOTE: It is not guaranteed that the variables idx are processed in non-decreasing order.
    # This is why we use bisect.insort, again, to ensure reproducibility.
    for _, is_duplicate, idx in minima.values():
        assert (
            not is_duplicate
        ), "Could not select the slice deterministically for one of the patients."
        bisect.insort(indices, idx)

    # NOTE: The variable indices is obtained deterministically from the list dataset.sample_identifiers.
    # We assume that the dataset.sample_identifiers is obtained deterministically as long as the dataset files are valid.
    dataset: torch.utils.data.Subset = torch.utils.data.Subset(dataset, indices)

    # Split the dataset into training and testing sets.
    train_split_size = 900
    test_split_size = 110
    assert (
            len(dataset) == train_split_size + test_split_size
    ), f"Dataset size {len(dataset)} does not match the expected size {train_split_size + test_split_size}."

    # Get random splits.
    generator = torch.Generator().manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_split_size, test_split_size], generator=generator
    )

    physics = get_physics()

    dinv.datasets.generate_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        physics=physics,
        save_dir="./LIDC_IDRI-Tomography",
        batch_size=1,
        device="cpu",
        show_progress_bar=True,
    )
