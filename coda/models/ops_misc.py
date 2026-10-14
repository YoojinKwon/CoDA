"""
1204
- Basic modules & scheduler for MobileNet_v2 and MobileNet_v3
- Refer to https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import warnings
from models.resnet import QConv2d
from models.quant_lq_bp_multi import get_extra_q_bits
from models._utils import _make_divisible
from typing import Callable, Optional, Sequence, Tuple, Union
from models._utils import _make_ntuple

from math import pi, cos, log, floor
from torch.optim.lr_scheduler import _LRScheduler


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

class CosineWarmupLR(_LRScheduler):
    '''
    Cosine lr decay function with warmup.
    Ref: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/optimizer/lr_scheduler.py
         https://github.com/Randl/MobileNetV3-pytorch/blob/master/cosine_with_warmup.py
    Lr warmup is proposed by 
        `Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`
    Cosine decay is proposed by 
        `Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`
    Args:
        optimizer (Optimizer): optimizer of a model.
        iter_in_one_epoch (int): number of iterations in one epoch.
        epochs (int): number of epochs to train.
        lr_min (float): minimum(final) lr.
        warmup_epochs (int): warmup epochs before cosine decay.
        last_epoch (int): init iteration. In truth, this is last_iter
    Attributes:
        niters (int): number of iterations of all epochs.
        warmup_iters (int): number of iterations of all warmup epochs.
        cosine_iters (int): number of iterations of all cosine epochs.
    '''

    def __init__(self, optimizer, epochs, iter_in_one_epoch, lr_min=0, warmup_epochs=0, last_epoch=-1):
        self.lr_min = float(lr_min)  # float로 고정
        self.niters = epochs * iter_in_one_epoch
        self.warmup_iters = iter_in_one_epoch * warmup_epochs
        self.cosine_iters = iter_in_one_epoch * (epochs - warmup_epochs)
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [(self.lr_min + (base_lr - self.lr_min) * self.last_epoch / self.warmup_iters) for base_lr in self.base_lrs]
        else:
            return [(self.lr_min + (base_lr - self.lr_min) * (1 + cos(pi * (self.last_epoch - self.warmup_iters) / self.cosine_iters)) / 2) for base_lr in self.base_lrs]






"""
MobileNet_v2
"""
class QConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = QConv2d,  # Default to QConv2d
        q_scheme=None,  # Quantization scheme
        q_bits_w=None,  # Quantization bits for weights
        q_bits_a=None,  # Quantization bits for activations
    ) -> None:

        # Automatically calculate padding if not provided
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        # Layers list
        layers = [
            conv_layer(
                q_scheme=q_scheme,  # Pass quantization scheme
                q_bits_w=q_bits_w,  # Pass weight quantization bits
                q_bits_a=q_bits_a,  # Pass activation quantization bits
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        # Add normalization layer if provided
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        # Add activation layer if provided
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        # Initialize the Sequential module
        super().__init__(*layers)
        # _log_api_usage_once(self)
        self.out_channels = out_channels

        # Warn users not to use this class directly
        if self.__class__ == QConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly; please use Conv2dNormActivation and similar classes."
            )
 
class QConv2dNormActivation(QConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        q_scheme=None,
        q_bits_w=None,
        q_bits_a=None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            dilation=dilation,
            inplace=inplace,
            bias=bias,
            conv_layer=QConv2d,  # Use QConv2d by default
            q_scheme=q_scheme,
            q_bits_w=q_bits_w,
            q_bits_a=q_bits_a,
        )
        
        
        
"""
MobileNet_v3        
"""
class QSqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
        q_scheme: Optional[str] = None,
        q_bits_w: Optional[int] = None,
        q_bits_a: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = QConv2d(q_scheme, q_bits_w, q_bits_a, input_channels, squeeze_channels, 1)
        self.fc2 = QConv2d(q_scheme, q_bits_w, q_bits_a, squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input

 
