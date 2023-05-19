import torch
import torch.nn as nn
from typing import Tuple


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential"""

    def forward(self, *args):
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """repeat module N times

    :param int N: repeat time
    :param function fn: function to generate module
    :return: repeated loss
    :rtype: MultiSequential
    """
    return MultiSequential(*[fn(n) for n in range(N)])

class LayerNorm(torch.nn.LayerNorm):
    """
    Layer normalization core.
    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1,  elementwise_affine=True):
        """
        Construct an LayerNorm object.
        """
        super(LayerNorm, self).__init__(nout, eps=1e-12, elementwise_affine=elementwise_affine)
        self.dim = dim

    def forward(self, x):
        """
        Apply layer normalization.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed to replace positionwise feed-forward network
    in Transforner block, which is introduced in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    Args:
        in_chans (int): Number of input channels.
        hidden_chans (int): Number of hidden channels.
        kernel_size (int): Kernel size of conv1d.
        dropout_rate (float): Dropout rate.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(
        self, in_chans: int, hidden_chans: int, kernel_size: int, dropout_rate: float
    ):
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans, in_chans, 1, stride=1, padding=(1 - 1) // 2
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, *, in_chans).

        Returns:
            Tensor: Batch of output tensors (B, *, hidden_chans)

        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Swish(torch.nn.Module):
    """
    Construct an Swish activation function for Conformer.
    """

    def forward(self, x):
        """
        Return Swish activation function.
        """
        return x * torch.sigmoid(x)

class ConvolutionModule(nn.Module):
    """
    ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        super(ConvolutionModule, self).__init__()
        # kernel_size should be an odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias, )
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=bias, )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias, )
        self.activation = activation

    def forward(self, x):
        """
        Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)