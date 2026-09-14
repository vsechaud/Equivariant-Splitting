import torch
from deepinv.physics.generator.base import PhysicsGenerator


def bernoulli_mask(B, W, split_ratio, device='cpu', rng=None):
    """
    Generate a column-wise binary Bernoulli mask (identical across B, C, H for each W).

    Args:
        shape (tuple): (B, C, H, W)
        split_ratio (float): probability of a 1 per column
        device (str): 'cpu' or 'cuda'
        seed (int or None): random seed for reproducibility

    Returns:
        torch.Tensor: binary mask of shape (B, C, H, W)
    """
    column_mask = torch.bernoulli(
        torch.full((B, W), split_ratio, device=device),
        generator=rng
    )

    return column_mask


class MaskTomography(PhysicsGenerator):
    def __init__(self, img_size, split_ratio=0.6, device='cpu', dtype=torch.float32, rng=None):
        super().__init__(device=device, dtype=dtype, rng=rng)
        self.img_size = img_size
        self.split_ratio = split_ratio
        self.dtype = dtype

    def step(self, batch_size=1, seed=None, **kwargs):
        self.rng_manual_seed(seed)

        C, H, W = self.img_size[-3:]

        column_mask = bernoulli_mask(batch_size, W, split_ratio=self.split_ratio, device=self.device, rng=self.rng).to(self.dtype)

        mask = column_mask[:, None, None, :].expand(batch_size, C, H, W)

        return {"mask": mask}
