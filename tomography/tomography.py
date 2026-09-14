from __future__ import annotations
from typing import Iterable
from types import MappingProxyType
from warnings import warn
import math

from numpy import ndarray
import torch

from deepinv.physics.forward import LinearPhysics, adjoint_function
from radon import (
    Radon,
    IRadon,
    RampFilter,
    ApplyRadon,
)


class Tomography(LinearPhysics):
    r"""
    (Computed) Tomography operator.

    The Radon transform is the integral transform which takes a square image :math:`x` defined on the plane to a function
    :math:`y=\forw{x}` defined on the (two-dimensional) space of lines in the plane, whose value at a particular line is equal
    to the line integral of the function over that line.

    .. note::

        The pseudo-inverse is computed using the filtered back-projection algorithm with a Ramp filter.
        This is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation which is
        robust to noise.

    .. note::

        The measurements are not normalized by the image size, thus the norm of the operator depends on the image size.

    .. note::

        This operator only handles 2D images. For more advanced use-cases, see the :class:`deepinv.physics.TomographyWithAstra` operator which handles 2D and 3D geometries.

    .. warning::

        The adjoint operator has small numerical errors due to interpolation. Set ``adjoint_via_backprop=True`` if you want to use the exact adjoint (computed via autograd).

    .. warning::

        By default, ``normalize`` is set to ``True`` if not specified. Initializing the operator without specifying the normalization behavior will issue a warning. Note that normalizing the operator affects the reconstruction dynamics, which may not always be suitable for real-world applications.

    :param int, torch.Tensor angles: These are the tomography angles. If the type is ``int``, the angles are sampled uniformly between 0 and 360 degrees.
        If the type is :class:`torch.Tensor`, the angles are the ones provided (e.g., ``torch.linspace(0, 180, steps=10)``).
    :param int img_width: width/height of the square image input.
    :param bool circle: If ``True`` both forward and backward projection will be restricted to pixels inside a circle
        inscribed in the square image.
    :param bool parallel_computation: if ``True``, all projections are performed in parallel. Requires more memory but is faster on GPUs.
    :param bool adjoint_via_backprop: if ``True``, the adjoint will be computed via :func:`deepinv.physics.adjoint_function`. Otherwise the inverse Radon transform is used.
        The inverse Radon transform is computationally cheaper (particularly in memory), but has a small adjoint mismatch.
        The backprop adjoint is the exact adjoint, but might break random seeds since it backpropagates through :func:`torch.nn.functional.grid_sample`, see the note `here <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html>`_.
    :param bool fbp_interpolate_boundary: the :func:`filtered back-projection <deepinv.physics.Tomography.A_dagger>` usually contains streaking artifacts on the boundary due to padding. For ``fbp_interpolate_boundary=True``
        these artifacts are corrected by cutting off the outer two pixels of the FBP and recovering them by interpolating the remaining image. This option
        only makes sense if ``circle`` is set to ``False``. Hence it will be ignored if ``circle`` is True.
    :param bool normalize: If ``True`` :func:`A <deepinv.physics.Tomography.A>` and :func:`A_adjoint <deepinv.physics.Tomography.A_adjoint>` are normalized so that the operator has unit norm. (default: ``True``)
    :param bool fan_beam: If ``True``, use fan beam geometry, if ``False`` use parallel beam
    :param dict[str, int | float] fan_parameters: Only used if fan_beam is ``True``. Contains the parameters defining the scanning geometry. The dict should contain the keys:

        - "pixel_spacing" defining the distance between two pixels in the image, default: 0.5 / (in_size)

        - "source_radius" distance between the x-ray source and the rotation axis (middle of the image), default: 57.5

        - "detector_radius" distance between the x-ray detector and the rotation axis (middle of the image), default: 57.5

        - "n_detector_pixels" number of pixels of the detector, default: 258

        - "detector_spacing" distance between two pixels on the detector, default: 0.077

        The default values are adapted from the geometry in :footcite:t:`khalil2023hyperspectral`.
        where pixel spacing, source and detector radius and detector spacing are given in cm.
        Note that a to small value of n_detector_pixels*detector_spacing can lead to severe circular artifacts in any reconstruction.
    :param str device: gpu or cpu.

    |sep|

    :Examples:

        Tomography operator with defined angles for 3x3 image:

        >>> from deepinv.physics import Tomography
        >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 4, 4)  # Define random 4x4 image
        >>> angles = torch.linspace(0, 45, steps=3)
        >>> physics = Tomography(angles=angles, img_width=4, circle=True, normalize=False)
        >>> physics(x)
        tensor([[[[ 0.0000, -0.1791, -0.1719],
                  [-0.5713, -0.4521, -0.5177],
                  [ 0.0340,  0.1448,  0.2334],
                  [ 0.0000, -0.0448, -0.0430]]]])

        Tomography operator with 3 uniformly sampled angles in [0, 360] for 3x3 image:

        >>> from deepinv.physics import Tomography
        >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 4, 4)  # Define random 4x4 image
        >>> physics = Tomography(angles=3, img_width=4, circle=True, normalize=False)
        >>> physics(x)
        tensor([[[[ 0.0000, -0.1806,  0.0500],
                  [-0.5713, -0.6076, -0.6815],
                  [ 0.0340,  0.3175,  0.0167],
                  [ 0.0000, -0.0452,  0.0989]]]])


    """

    def __init__(
            self,
            angles: int | Iterable[float],
            img_width: int,
            circle: bool = False,
            parallel_computation: bool = True,
            adjoint_via_backprop: bool = True,
            fbp_interpolate_boundary: bool = False,
            normalize: bool | None = None,
            fan_beam: bool = False,
            fan_parameters: dict[str, int | float] = None,
            device: torch.device | str = torch.device("cpu"),
            dtype: torch.dtype = torch.float,
            **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(angles, int):
            theta = torch.linspace(0, 180, steps=angles + 1, device=device)[:-1].to(
                device
            )
        elif isinstance(angles, (list, tuple, ndarray)):
            theta = torch.tensor(angles).to(device)
        elif isinstance(angles, torch.Tensor):
            theta = angles
        else:
            raise ValueError(
                f"angles must be int, float, iterable or Tensor, but got {type(angles)}"
            )

        self.register_buffer("theta", theta)

        self.fan_beam = fan_beam
        self.adjoint_via_backprop = adjoint_via_backprop
        if fan_beam or adjoint_via_backprop:
            self._auto_grad_adjoint_fn = None
            self._auto_grad_adjoint_input_shape = (1, 1, img_width, img_width)
        if circle and fbp_interpolate_boundary:
            # interpolate boundary does not make sense if circle is True
            warn(
                "The argument fbp_interpolate_boundary=True is not applicable if circle=True. The value fbp_interpolate_boundary will be changed to False..."
            )
            fbp_interpolate_boundary = False
        self.fbp_interpolate_boundary = fbp_interpolate_boundary
        self.img_width = img_width
        self.device = device
        self.dtype = dtype
        self.radon = Radon(
            img_width,
            theta,
            circle=circle,
            parallel_computation=parallel_computation,
            fan_beam=fan_beam,
            fan_parameters=fan_parameters,
            device=device,
            dtype=dtype,
        ).to(device)
        if not self.fan_beam:
            self.iradon = IRadon(
                img_width,
                theta,
                circle=circle,
                parallel_computation=parallel_computation,
                device=device,
                dtype=dtype,
            ).to(device)
        else:
            self.filter = RampFilter(dtype=dtype, device=device)

        if normalize is None:
            warn(
                "The default value of `normalize` is not specified and will be automatically set to `True`. Set `normalize` explicitly to `True` or `False` to avoid this warning.",
            )
            normalize = True

        self.normalize = False
        if normalize:
            operator_norm = self.compute_norm(
                torch.randn(
                    (img_width, img_width),
                    generator=torch.Generator(self.device).manual_seed(0),
                    device=self.device,
                )[None, None],
            ).sqrt()
            self.register_buffer("operator_norm", operator_norm)
            self.normalize = True

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward projection.

        :param torch.Tensor x: input of shape [B,C,H,W]
        :return: measurement of shape [B,C,A,N], with A the number of angular positions, and N the number of detector cells.
        """
        if self.fan_beam or self.adjoint_via_backprop:
            output = self.radon(x)
        else:
            output = ApplyRadon.apply(x, self.radon, self.iradon, False)
        if self.normalize:
            output = output / self.operator_norm

        return output

    def fbp(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes the filtered back-projection (FBP) of the measurements.

        .. tip::

            By default, the FBP reconstruction can display artifacts at the borders. Set ``fbp_interpolate_boundary=True`` to remove them with padding.


        :param torch.Tensor y: measurements of shape [B,C,A,N], with A the number of angular positions, and N the number of detector cells
        :return: filtered back-projection of shape [B,C,H,W]
        """
        if self.fan_beam or self.adjoint_via_backprop:
            if self.fan_beam:
                y = self.filter(y)
            else:
                y = self.iradon.filter(y)
            output = (
                    self.A_adjoint(y, **kwargs) * torch.pi / (2 * len(self.radon.theta))
            )
            if self.normalize:
                output = output * self.operator_norm**2
        else:
            y = self.iradon.filter(y)
            output = (
                    ApplyRadon.apply(y, self.radon, self.iradon, True)
                    * torch.pi
                    / (2 * len(self.iradon.theta))
            )
            if self.normalize:
                output = output * self.operator_norm

        if self.fbp_interpolate_boundary:
            output = output[:, :, 2:-2, 2:-2]
            output = torch.nn.functional.pad(output, (2, 2, 2, 2), mode="replicate")
        return output

    def A_dagger(self, y: torch.Tensor, fbp: bool = False, **kwargs) -> torch.Tensor:
        r"""
        Computes the solution in :math:`x` to :math:`y = Ax` using a least squares solver. A faster approximation can be obtained by setting ``fbp=True``, which computes the filtered back-projection of the measurements.

        .. warning::

            The filtered back-projection algorithm is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation that is robust to noise.

        :param torch.Tensor y: measurements of shape [B,C,A,N], with A the number of angular positions, and N the number of detector cells
        :return: filtered back-projection of shape [B,C,H,W]
        """
        if fbp:
            return self.fbp(y, **kwargs)
        else:
            return super(Tomography, self).A_dagger(y, **kwargs)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes adjoint of the tomography operator.

        .. warning::

            The default adjoint operator has small numerical errors due to interpolation. Set ``adjoint_via_backprop=True`` if you want to use the exact adjoint (computed via autograd).

        :param torch.Tensor y: measurements of shape [B,C,A,N]
        :return: scaled back-projection of shape [B,C,H,W]
        """
        if self.fan_beam or self.adjoint_via_backprop:
            # lazy implementation for the adjoint...
            if (
                    self._auto_grad_adjoint_fn is None
                    or self._auto_grad_adjoint_input_shape
                    != (y.size(0), y.size(1), self.img_width, self.img_width)
            ):
                self._auto_grad_adjoint_fn = adjoint_function(
                    self.A,
                    (y.shape[0], y.shape[1], self.img_width, self.img_width),
                    device=self.device,
                    dtype=self.dtype,
                )
                self._auto_grad_adjoint_input_shape = (
                    y.size(0),
                    y.size(1),
                    self.img_width,
                    self.img_width,
                )

            output = self._auto_grad_adjoint_fn(y)
        else:
            output = ApplyRadon.apply(y, self.radon, self.iradon, True)

        if self.normalize:
            output = output / self.operator_norm

        return output
