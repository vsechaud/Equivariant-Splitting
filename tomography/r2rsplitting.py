import math

import torch
import deepinv as dinv
from deepinv.loss import SplittingLoss


# A R2R-like loss to be used in conjunction with the splitting loss
class SplitR2RLoss(dinv.loss.R2RLoss):

    def __init__(self, mask_generator, noise_model, alpha=0.15, split_ratio=0.6, weight=1.0, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.split_ratio = split_ratio
        self.noise_model = noise_model
        self.noise_model.update_parameters(sigma=noise_model.sigma * math.sqrt(alpha / (1 - alpha)))
        self.weight = weight
        self.mask_generator = mask_generator


    def forward(self, x_net, y, physics, model, **kwargs):
        ya = model.get_corruption()
        yb =  (y - ya*(1-self.alpha)) / self.alpha

        mask = model.get_mask() * getattr(physics, "mask", 1.0)
        r2rloss = self.metric(mask * physics.A(x_net), mask * yb)
        return  self.weight * r2rloss / mask.mean()

    def adapt_model(self, model):
        return (
            model
            if isinstance(model, self.R2RSplittingModel)
            else self.R2RSplittingModel(
                model, split_ratio=self.split_ratio, mask_generator=self.mask_generator, noise_model=self.noise_model, eval_n_samples=self.eval_n_samples
            )
        )

    class R2RSplittingModel(SplittingLoss.SplittingModel):
        def __init__(self, model, split_ratio, mask_generator, noise_model, eval_n_samples):
            super().__init__(
                model,
                split_ratio=split_ratio,
                mask_generator=mask_generator,
                eval_n_samples=eval_n_samples,
                eval_split_input=True,
                eval_split_output=False,
                pixelwise=True,
            )
            self.noise_model = noise_model

        def split(self, mask, y, physics=None):
            y1, physics1 = SplittingLoss.split(mask, y, physics)
            noiser_y1 = self.noise_model(y1)
            self.corruption = noiser_y1
            return mask * noiser_y1, physics1

        def get_corruption(self):
            return self.corruption

