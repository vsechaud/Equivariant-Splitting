from deepinv.loss.metric.metric import Metric
from deepinv.physics.forward import LinearPhysics
from deepinv.transform.base import Transform
from deepinv.models.base import Reconstructor

import torch


class EQUIV(Metric):

    def __init__(
        self,
        *,
        transform: Transform,
        metric: Metric,
        n_samples: int = 5,
        db: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._transform = transform
        self._metric_base = metric
        self._n_samples = n_samples
        self._db = db

    def metric(self, x, x_net, y, physics, model, **kwargs) -> torch.Tensor:
        return self._metric_fn(
            y=y,
            x_net=x_net,
            model=model,
            transform=self._transform,
            physics=physics,
            metric=self._metric_base,
            n_samples=self._n_samples,
            db=self._db,
        )

    @staticmethod
    def _metric_fn(
        *,
        y: torch.Tensor,
        x_net: torch.Tensor,
        model: Reconstructor,
        transform: Transform,
        physics: LinearPhysics,
        metric: Metric,
        n_samples: int,
        db: bool,
    ):
        n_trans = transform.n_trans
        if n_trans != 1:
            raise ValueError(
                f"Unsupported value of Transform.n_trans, f{n_trans}, should be 1"
            )
        model_t = transform.symmetrize(model)
        out = None
        for _ in range(n_samples):
            x_net_t = model_t(y, physics)
            values = metric(
                x=x_net,
                x_net=x_net_t,
                y=None,
                physics=None,
                model=None,
            )
            if out is None:
                out = values
            else:
                out += values
        out /= n_samples
        if db:
            out = -10 * torch.log10(out)
        return out
