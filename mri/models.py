import torch
import torch.nn as nn
from deepinv.models.drunet import test_pad
from deepinv.models.base import Denoiser, Reconstructor
from deepinv.transform import Transform, Rotate, Reflect


# Adapted from
# https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/layer_norm.py#L50
class LayerNorm_AF(nn.Module):
    """
    Alias-Free Layer Normalization

    From `"Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations" by Michaeli et al. <https://doi.org/10.48550/arXiv.2303.08085>`_

    :param int in_channels: number of input channels.
    :param int out_channels: number of output channels.
    :param bool residual: if True, the output is the sum of the input and the denoised image.
    :param bool cat: if True, the network uses skip connections.
    :param int scales: number of scales in the network.
    :param bool rotation_equivariant: if True, the network is rotation-equivariant.

    :param Union[int, list, torch.Size] normalized_shape: Input shape from an expected input of size.
    :param float eps: A value added to the denominator for numerical stability. Default: 1e-6.
    :param bool bias: If set to False, the layer will not learn an additive bias. Default: True.
    """

    def __init__(self, normalized_shape, eps=1e-6, bias=True, type="AF"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.bias = None
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        # In our previous implementation, self.u_dims was wrongly set to (1, 2, 3) but it is actually meant to be equal to 1
        # in the original implementation.
        # https://github.com/hmichaeli/alias_free_convnets/blob/7ef0a6eea3990c015d746d13a5cd79735809dd45/models/layer_norm.py#L16
        if type == "AF":
            self.u_dims = 1
        elif type == "CHW":
            self.u_dims = (1, 2, 3)
        else:
            raise ValueError(f"type={type} is not supported")
        self.s_dims = (1, 2, 3)

    def forward(self, x):
        """
        Forward pass for layer normalization.

        :param torch.Tensor x: Input tensor
        """
        u = x.mean(self.u_dims, keepdim=True)
        s = (x - u).pow(2).mean(self.s_dims, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x
        if self.bias is not None:
            x = x + self.bias[:, None, None]
        return x


class BFBatchNorm2d(nn.BatchNorm2d):
    r"""
    From Mohan et al.

    "Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks"
    S. Mohan, Z. Kadkhodaie, E. P. Simoncelli, C. Fernandez-Granda
    Int'l. Conf. on Learning Representations (ICLR), Apr 2020.
    """

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, use_bias=False, affine=True
    ):
        super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)
        self.use_bias = use_bias
        self.affine = affine

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)
        if self.training is not True:
            if self.use_bias:
                y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1) ** 0.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (
                            1 - self.momentum
                        ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1) ** 0.5 + self.eps)
        if self.affine:
            y = self.weight.view(-1, 1) * y
            if self.use_bias:
                y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0, 1)


class UNet(Denoiser):
    r"""
    U-Net convolutional denoiser.

    This network is a fully convolutional denoiser based on the U-Net architecture. The number of downsample steps
    can be controlled with the ``scales`` parameter. The number of trainable parameters increases with the number of
    scales.

    .. warning::
        When using the bias-free batch norm ``BFBatchNorm2d`` via ``batch_norm="biasfree"``, NaNs may be encountered
        during training, causing the whole training procedure to fail.

    :param int in_channels: input image channels
    :param int out_channels: output image channels
    :param bool residual: use a skip-connection between output and output.
    :param bool circular_padding: circular padding for the convolutional layers.
    :param bool cat: use skip-connections between intermediate levels.
    :param bool bias: use learnable biases.
    :param bool, str batch_norm: if False, no batchnorm applied, if ``True``, use :meth:`torch.nn.BatchNorm2d`,
        if ``batch_norm="biasfree"``, use ``BFBatchNorm2d`` from
        `"Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks" by Mohan et al. <https://arxiv.org/abs/1906.05478>`_.
    :param int scales: Number of downsampling steps used in the U-Net. The options are 2,3,4 and 5.
        The number of trainable parameters increases with the scale.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        residual=True,
        circular_padding=False,
        cat=True,
        bias=True,
        batch_norm=True,
        scales=4,
        norm_type="layer_norm_af",
    ):
        super(UNet, self).__init__()
        self.name = "unet"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.compact = scales
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        biasfree = batch_norm == "biasfree"

        def norm(ch):
            if norm_type == "batch_norm":
                return (
                    BFBatchNorm2d(ch, use_bias=bias) if biasfree else nn.BatchNorm2d(ch)
                )
            elif norm_type == "layer_norm_af":
                return LayerNorm_AF(ch, bias=bias)
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")

        def conv_block(ch_in, ch_out):
            if batch_norm:
                return nn.Sequential(
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    norm(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        ch_out,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    norm(ch_out),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        ch_out,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    nn.ReLU(inplace=True),
                )

        def up_conv(ch_in, ch_out):
            if batch_norm:
                return nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    norm(ch_out),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    nn.ReLU(inplace=True),
                )

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = (
            conv_block(ch_in=128, ch_out=256) if self.compact in [3, 4, 5] else None
        )
        self.Conv4 = (
            conv_block(ch_in=256, ch_out=512) if self.compact in [4, 5] else None
        )
        self.Conv5 = conv_block(ch_in=512, ch_out=1024) if self.compact in [5] else None

        self.Up5 = up_conv(ch_in=1024, ch_out=512) if self.compact in [5] else None
        self.Up_conv5 = (
            conv_block(ch_in=1024, ch_out=512) if self.compact in [5] else None
        )

        self.Up4 = up_conv(ch_in=512, ch_out=256) if self.compact in [4, 5] else None
        self.Up_conv4 = (
            conv_block(ch_in=512, ch_out=256) if self.compact in [4, 5] else None
        )

        self.Up3 = up_conv(ch_in=256, ch_out=128) if self.compact in [3, 4, 5] else None
        self.Up_conv3 = (
            conv_block(ch_in=256, ch_out=128) if self.compact in [3, 4, 5] else None
        )

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if self.compact == 5:
            self._forward = self.forward_standard
        if self.compact == 4:
            self._forward = self.forward_compact4
        if self.compact == 3:
            self._forward = self.forward_compact3
        if self.compact == 2:
            self._forward = self.forward_compact2

    def forward(self, x, sigma=None, **kwargs):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """

        factor = 2 ** (self.compact - 1)
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward(x)
        else:
            return test_pad(self._forward, x, modulo=factor)

    def forward_standard(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        if self.cat:
            d5 = torch.cat((x4, d5), dim=cat_dim)
            d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if self.cat:
            d4 = torch.cat((x3, d4), dim=cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out

    def forward_compact4(self, x):
        # def forward_compact4(self, x):
        # encoding path
        cat_dim = 1
        input = x

        x1 = self.Conv1(input)  # 1->64

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # 64->128

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # 128->256

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # 256->512

        d4 = self.Up4(x4)  # 512->256
        if self.cat:
            d4 = torch.cat((x3, d4), dim=cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)  # 256->128
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)  # 128->64
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out

    def forward_compact3(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d3 = self.Up3(x3)
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out

    def forward_compact2(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        d2 = self.Up2(x2)
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out


class EquivariantReconstructor(Reconstructor):
    r"""
    Turns the reconstructor model into an equivariant reconstructor with respect to geometric transforms.

    Recall that a reconstructor is equivariant with respect to a group of transformations if it commutes with the action of
    the group. More precisely, let :math:`\mathcal{G}` be a group of transformations :math:`\{T_g\}_{g\in \mathcal{G}}`
    and :math:`\inversename` a reconstruction model. Then, :math:`\inversename` is equivariant with respect to :math:`\mathcal{G}`
    if :math:`\inversef{y,AT_g} = T_g\inversef{y,A}` for any measurement :math:`y` and any :math:`g\in \mathcal{G}`.

    The reconstruction model can be turned into an equivariant denoiser by averaging over the group of transforms, i.e.

    .. math::
        \operatorname{R}^{\text{eq}}(y,A) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g(\inversef{y}{AT_g}).

    Otherwise, as proposed in https://arxiv.org/abs/2312.01831, a Monte Carlo approximation can be obtained by
    sampling :math:`g \sim \mathcal{G}` at random and applying

    .. math::
        \operatorname{R}^{\text{MC}}(y,A) = T_g(\inversef{y}{AT_g}).

    .. note::

        We have implemented many popular geometric transforms, see :ref:`docs <transform>`. You can set the number of Monte Carlo samples by passing ``n_trans``
        into the transforms, for example ``Rotate(n_trans=2)`` will average over 2 samples per call. For rotate and reflect, by setting ``n_trans``
        to the maximum (e.g. 4 for 90 degree rotations, 2 for 1D reflections), it will average over the whole group, for example:

        ``Rotate(n_trans=4, multiples=90, positive=True) * Reflect(n_trans=2, dims=[-1])``

    See :ref:`sphx_glr_auto_examples_basics_demo_transforms.py` for an example.

    :param Callable model: Reconstruction model :math:`\inversef{y}{A}`.
    :param Transform transform: geometric transformation. If None, defaults to rotations of multiples of 90 with horizontal flips (see note above).
        See :ref:`docs <transform>` for list of available transforms.
    :param bool random: if True, the model is applied to a randomly transformed version of the input image
        each time i.e. a Monte-Carlo approximation of an equivariant denoiser.
        If False, the model is applied to the average of all the transformed images, turning the reconstructor into an
        equivariant reconstructor with respect to the chosen group of transformations. Ignored if ``transform`` is provided.
    """

    def __init__(self, model: Reconstructor, random=False, eval_mode: str = "same"):
        super().__init__()
        self.model = model

        single_transform = Rotate(n_trans=1, multiples=90, positive=True) * Reflect(
            n_trans=1, dim=[-1]
        )
        full_group_transforms = Rotate(
            n_trans=4, multiples=90, positive=True
        ) * Reflect(n_trans=2, dim=[-1])
        if random:
            self.transform = single_transform
        else:
            self.transform = full_group_transforms
        if eval_mode == "same":
            self.eval_transform = self.transform
        elif eval_mode == "full":
            self.eval_transform = full_group_transforms
        else:
            raise ValueError(
                f"eval_mode {eval_mode} not recognized, should be 'same' or 'full'"
            )

    def forward(self, y, physics, *reconstructor_args, **reconstructor_kwargs):
        r"""
        Symmetrize the reconstructor by the transformation to create an equivariant reconstructor and apply to input.

        The symmetrization collects the average if multiple samples are used (controlled with ``n_trans`` in the transform).

        :param torch.Tensor x: input image.
        :param \*denoiser_args: args for denoiser function e.g. sigma noise level.
        :param \**denoiser_kwargs: kwargs for denoiser function e.g. sigma noise level.
        :return: denoised image.
        """
        if self.training:
            transform = self.transform
        else:
            transform = self.eval_transform
        return transform.symmetrize(self.model, average=True)(
            y, physics, *reconstructor_args, **reconstructor_kwargs
        )
