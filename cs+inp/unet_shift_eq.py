import torch
import torch.nn as nn
from .drunet import test_pad
from .base import Denoiser


import torch.nn.functional as F
from torch import Tensor
import numpy as np

def bicubic_filter(factor=2):
    r"""
    Bicubic filter.

    It has size (4*factor, 4*factor) and is defined as

    .. math::

        \begin{equation*}
            w(x, y) = \begin{cases}
                (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{if } |x| \leq 1 \\
                a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{if } 1 < |x| < 2 \\
                0 & \text{otherwise}
            \end{cases}
        \end{equation*}

    for :math:`x, y \in {-2\text{factor} + 0.5, -2\text{factor} + 0.5 + 1/\text{factor}, \ldots, 2\text{factor} - 0.5}`.

    :param int factor: downsampling factor
    """
    x = np.arange(start=-2 * factor + 0.5, stop=2 * factor, step=1) / factor
    a = -0.5
    x = np.abs(x)
    w = ((a + 2) * np.power(x, 3) - (a + 3) * np.power(x, 2) + 1) * (x <= 1)
    w += (
        (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a)
        * (x > 1)
        * (x < 2)
    )
    w = np.outer(w, w)
    w = w / np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)


def conv2d(
    x: Tensor, filter: Tensor, padding: str = "valid", correlation=False
) -> torch.Tensor:
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    if not correlation:
        filter = torch.flip(filter, [-2, -1])

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()

    if c != C:
        assert c == 1
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert b == 1
        filter = filter.expand(B, -1, -1, -1)

    if padding != "valid":
        ph = h // 2
        ih = (h - 1) % 2
        pw = w // 2
        iw = (w - 1) % 2
        pad = (pw, pw - iw, ph, ph - ih)  # because functional.pad is w,h instead of h,w

        x = F.pad(x, pad, mode=padding, value=0)
        B, C, H, W = x.size()

    # Move batch dim of the input into channels
    x = x.reshape(1, -1, H, W)
    # Expand the channel dim of the filter and move it into batch dimension
    filter = filter.reshape(B * C, -1, h, w)
    # Perform the convolution, using the groups parameter
    output = F.conv2d(x, filter, padding="valid", groups=B * C)
    # Make it in the good shape
    output = output.view(B, C, output.size(-2), -1)

    return output

class EquivMaxPool(nn.Module):
    r"""
    Max pooling layer that is equivariant to translations.

    :param int kernel_size: size of the pooling window.
    :param int stride: stride of the pooling operation.
    :param int padding: padding to apply before pooling.
    :param bool circular_padding: circular padding for the convolutional layers.
    """

    def __init__(self, antialias=True, factor=2, device='cpu'):
        super(EquivMaxPool, self).__init__()
        self.antialias = antialias
        if antialias:
            self.antialias_kernel = bicubic_filter(factor).to(device)

    def downscale(self, x):
        r"""
        Apply the equivariant max pooling.

        :param torch.Tensor x: input tensor.
        """

        if self.antialias:
            x = conv2d(x, self.antialias_kernel, padding='circular')

        B, _, _, _ = x.shape

        x1 = x[:, :, ::2, ::2].unsqueeze(0)
        x2 = x[:, :, ::2, 1::2].unsqueeze(0)
        x3 = x[:, :, 1::2, ::2].unsqueeze(0)
        x4 = x[:, :, 1::2, 1::2].unsqueeze(0)
        out = torch.cat([x1, x2, x3, x4], dim=0)  # (4, B, C, H/2, W/2)
        ind = torch.norm(out, dim=(2, 3, 4), p=2)  # (4, B)
        self.ind = torch.argmax(ind, dim=0)  # (B)
        out = out[self.ind, torch.arange(B), ...]  # (B, C, H/2, W/2)

        return out


    def upscale(self, x):
        B, C, H, W = x.shape

        out = torch.zeros((B, C, H * 2, W * 2), device=x.device)
        out[:, :, ::2, ::2] = x
        filt = torch.zeros((B, 1, 2, 2), device=x.device)
        filt[self.ind == 0, :, 0, 0] = 1
        filt[self.ind == 1, :, 0, 1] = 1
        filt[self.ind == 2, :, 1, 0] = 1
        filt[self.ind == 3, :, 1, 1] = 1
        out = conv2d(out, filt, padding="constant")

        if self.antialias:
            out = conv2d(out, self.antialias_kernel, padding='circular')

        return out


class BFBatchNorm2d(nn.BatchNorm2d):
    r"""
    From :footcite:t:`mohan2020robust`.
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

class LayerNorm_AF(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.bias = None
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.u_dims = 1
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

class UNetShiftEQ(Denoiser):
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
    :param bool, str batch_norm: if False, no batchnorm applied, if ``True``, use :class:`torch.nn.BatchNorm2d`,
        if ``batch_norm="biasfree"``, use ``BFBatchNorm2d`` from
        `"Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks" by Mohan et al. <https://arxiv.org/abs/1906.05478>`_.
    :param int scales: Number of downsampling steps used in the U-Net. The options are 2,3,4 and 5.
        The number of trainable parameters increases with the scale.
    :param bool layernorm_af: use layer normalization instead of batch normalization. Its value is ignored if batch_norm is False.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        residual=True,
        circular_padding=True,
        cat=True,
        bias=True,
        batch_norm=False,
        layernorm_af=True,
        scales=4,
        device="cpu",
    ):
        super(UNetShiftEQ, self).__init__()
        self.name = "unet"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.compact = scales

        biasfree = batch_norm == "biasfree"

        def norm(channels, bias):
            if layernorm_af:
                return LayerNorm_AF(channels, bias=bias)
            else:
                if biasfree:
                    return BFBatchNorm2d(channels, use_bias=bias)
                else:
                    return nn.BatchNorm2d(channels)

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
                    norm(channels=ch_out, bias=bias),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode="circular" if circular_padding else "zeros"
                    ),
                    norm(channels=ch_out, bias=bias),
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
                        ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode="circular" if circular_padding else "zeros"
                    ),
                    nn.ReLU(inplace=True),
                )

        def up_conv(ch_in, ch_out):
            if batch_norm:
                return nn.Sequential(
                    nn.Conv2d(
                        ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode="circular" if circular_padding else "zeros"
                    ),
                    norm(channels=ch_out, bias=bias),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(
                        ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode="circular" if circular_padding else "zeros"
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

        self.Maxpool1 = EquivMaxPool(device=device)

        self.Maxpool2 = EquivMaxPool(device=device)

        self.Maxpool3 = EquivMaxPool(device=device)

        self.Maxpool4 = EquivMaxPool(device=device)

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

        x2 = self.Maxpool1.downscale(x1)
        x2 = self.Conv2(x2)  # 64->128

        x3 = self.Maxpool2.downscale(x2)
        x3 = self.Conv3(x3)  # 128->256

        x4 = self.Maxpool3.downscale(x3)
        x4 = self.Conv4(x4)  # 256->512

        d4 = self.Maxpool3.upscale(x4)
        d4 = self.Up4(d4)  # 512->256
        if self.cat:
            d4 = torch.cat((x3, d4), dim=cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Maxpool2.upscale(d4)
        d3 = self.Up3(d3)  # 256->128
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Maxpool1.upscale(d3)
        d2 = self.Up2(d2)  # 128->64
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1 + x if self.residual and self.in_channels == self.out_channels else d1
        return out
