import torch.utils.data
import deepinv as dinv

from typing import Union

from tomography import Tomography

NOISE_SIGMA = 0.001

def get_physics(device: Union[str, torch.device] = "cpu") -> dinv.physics.Physics:
    physics = Tomography(
        angles=50,
        img_width=256,
        circle=False,
        parallel_computation=True,
        normalize=True,
        fan_beam=False,
        device=device,
    )
    physics.set_noise_model(dinv.physics.GaussianNoise(sigma=NOISE_SIGMA))
    return physics

