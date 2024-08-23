# coding=utf-8
# Copyright 2024 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import math
from typing import Optional, Sequence, Union, Dict, Tuple
import warnings

from torch import Tensor, nn
import torch

from ...activations import ACT2CLS
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_rtmdet import RTMDetCSPNeXtConfig

class RTMDetCSPNeXtPreTrainedModel(PreTrainedModel):
    config_class = RTMDetCSPNeXtConfig
    base_model_prefix = "rtmdet_cspnext"
    main_input_names = "pixel_values"

    def _init_weights(self, module):
        # TODO: init weights
        raise NotImplementedError()


def efficient_conv_bn_eval_forward(bn: nn.modules.batchnorm,
                                   conv: nn.modules.conv._ConvNd,
                                   x: torch.Tensor):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for training as well. It reduces memory and computation cost.

    Args:
        bn (_BatchNorm): a BatchNorm module.
        conv (nn._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """
    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    weight_coeff = torch.rsqrt(bn.running_var +
                               bn.eps).reshape([-1] + [1] *
                                               (len(conv.weight.shape) - 1))
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() *\
        (bias_on_the_fly - bn.running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)

def build_norm_layer(norm_cfg: dict, num_features: int):
    cfg = norm_cfg.copy()
    type = cfg.pop('type')

    if type == 'BN':
        return 'bn', nn.BatchNorm2d(num_features, **cfg)

    if type == 'SyncBN':
        return 'syncbn', nn.SyncBatchNorm(num_features, **cfg)

    raise NotImplementedError(f'Norm type {type} is not supported')

def build_activation_layer(act_cfg: dict):
    act_type = act_cfg.pop('type').lower()
    act_cls = ACT2CLS[act_type]
    return act_cls(**act_cfg)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 order: tuple = ('conv', 'norm', 'act')):
        super().__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.efficient_conv_bn_eval_forward = None

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        conv_padding = padding
        # build convolution layer
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(
                norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (nn.modules.batchnorm, nn.modules.instancenorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None  # type: ignore

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self,
                x: Tensor,
                activate: bool = True,
                norm: bool = True) -> Tensor:
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == 'conv':
                # if the next operation is norm and we have a norm layer in
                # eval mode and we have enabled `efficient_conv_bn_eval` for
                # the conv operator, then activate the optimized forward and
                # skip the next norm operator since it has been fused
                if layer_index + 1 < len(self.order) and \
                        self.order[layer_index + 1] == 'norm' and norm and \
                        self.with_norm and not self.norm.training and \
                        self.efficient_conv_bn_eval_forward is not None:
                    self.conv.forward = partial(
                        self.efficient_conv_bn_eval_forward, self.norm,
                        self.conv)
                    layer_index += 1
                    x = self.conv(x)
                    del self.conv.forward
                else:
                    x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
            layer_index += 1
        return x

class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``. Default: 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``. Default: 0.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Dict = dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        dw_norm_cfg = norm_cfg
        dw_act_cfg = act_cfg
        pw_norm_cfg =norm_cfg
        pw_act_cfg = act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,  # type: ignore
            act_cfg=dw_act_cfg,
            **kwargs)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class SPPFBottleneck(nn.Module):
    """Spatial pyramid pooling layer

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes: Union[int, Sequence[int]]=(5, 9, 13),
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)
        
        self.conv2 = ConvModule(
            conv2_in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x

class CSPNeXtBlock(nn.Module):
    """The basic bottleneck block used in CSPNeXt.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): Expand ratio of the hidden channel. Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out. Only works
            when in_channels == out_channels. Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        kernel_size (int): The kernel size of the second convolution layer.
            Defaults to 5.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 kernel_size: int = 5,
                 norm_cfg = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='SiLU')
                 ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out

class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        # if digit_version(torch.__version__) < (1, 7, 0):
        #     self.act = nn.Hardsigmoid()
        # else:
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        use_cspnext_block (bool): Whether to use CSPNeXt block.
            Defaults to False.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish')
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 channel_attention: bool = False,
                 norm_cfg = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='Swish'),
                 ) -> None:
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)

def _get_bases_name(m):
    return [b.__name__ for b in m.__class__.__bases__]

def update_init_info(module, init_info):
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.
    """
    assert hasattr(
        module,
        '_params_init_info'), f'Can not find `_params_init_info` in {module}'
    for name, param in module.named_parameters():

        assert param in module._params_init_info, (
            f'Find a new :obj:`Parameter` '
            f'named `{name}` during executing the '
            f'`init_weights` of '
            f'`{module.__class__.__name__}`. '
            f'Please do not add or '
            f'replace parameters during executing '
            f'the `init_weights`. ')

        # The parameter has been changed during executing the
        # `init_weights` of module
        mean_value = param.data.mean().cpu()
        if module._params_init_info[param]['tmp_mean_value'] != mean_value:
            module._params_init_info[param]['init_info'] = init_info
            module._params_init_info[param]['tmp_mean_value'] = mean_value

class KaimingInit():
    r"""Initialize module parameters with the values according to the method
    described in the paper below.

    `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification - He, K. et al. (2015).
    <https://www.cv-foundation.org/openaccess/content_iccv_2015/
    papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`_

    Args:
        a (int | float): the negative slope of the rectifier used after this
            layer (only used with ``'leaky_relu'``). Defaults to 0.
        mode (str):  either ``'fan_in'`` or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the
            magnitudes in the backwards pass. Defaults to ``'fan_out'``.
        nonlinearity (str): the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` .
            Defaults to 'relu'.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'`` or
            ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 distribution='normal',
                 **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution
        self.layer = ['Conv2d']
        self.bias = 0

    def __call__(self, module):

        def init(m):
            layername = m.__class__.__name__
            basesname = _get_bases_name(m)
            if len(set(self.layer) & set([layername] + basesname)):
                kaiming_init(m, self.a, self.mode, self.nonlinearity,
                                self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: a={self.a}, mode={self.mode}, ' \
               f'nonlinearity={self.nonlinearity}, ' \
               f'distribution ={self.distribution}, bias={self.bias}'
        return info

# From left to right:
# in_channels, out_channels, num_blocks, add_identity, use_spp
arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
act_cfg = dict(type='SiLU')

class RTMDetCSPNeXtEmbeddings(nn.Module):
    def __init__(self, config: RTMDetCSPNeXtConfig):
        super().__init__()

        arch_setting = arch_settings[config.arch]
        
        self.embedder = nn.Sequential(
            ConvModule(
                3,
                int(arch_setting[0][0] * config.widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * config.widen_factor // 2),
                int(arch_setting[0][0] * config.widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * config.widen_factor // 2),
                int(arch_setting[0][0] * config.widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        return self.embedder(x)

class RTMDetCSPNeXtStage(nn.Module):
    def __init__(self, config: RTMDetCSPNeXtConfig, in_channels: int, out_channels: int, num_blocks: int, add_identity: bool, use_spp: bool):
        super().__init__()
        in_channels = int(in_channels * config.widen_factor)
        out_channels = int(out_channels * config.widen_factor)
        num_blocks = max(round(num_blocks * config.deepen_factor), 1)
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.layers = nn.ModuleList([conv_layer])
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=config.spp_kernel_sizes,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.layers.append(spp)

        csp_layer = CSPLayer(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            expand_ratio=config.expand_ratio,
            channel_attention=config.channel_attention,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers.append(csp_layer)
    
    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

class RTMDetCSPNeXtEncoder(nn.Module):
    def __init__(self, config: RTMDetCSPNeXtConfig):
        super().__init__()
        self.stages = nn.ModuleList([])

        arch_setting = arch_settings[config.arch]

        for i, (in_channels, out_channels, num_blocks, add_identity, use_ssp) in enumerate(arch_setting):
            stage = RTMDetCSPNeXtStage(config, in_channels, out_channels, num_blocks, add_identity, use_ssp)
            self.stages.append(stage)

    def forward(self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class RTMDetCSPNeXtBackbone(RTMDetCSPNeXtPreTrainedModel, BackboneMixin):
    def __init__(self, config: RTMDetCSPNeXtConfig):
        super().__init__(config)
        # TODO?!?
        #KaimingInit(a = math.sqrt(5), distribution='uniform', mode='fan_in', nonlinearity='leaky_relu')(self)
        super()._init_backbone(config)
        
        self.config = config

        self.embedder = RTMDetCSPNeXtEmbeddings(config)
        self.encoder = RTMDetCSPNeXtEncoder(config)
        
    def forward(self, 
                pixel_valuesixel_values: Tensor, 
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None
    ) -> BackboneOutput:

        embedding_output = self.embedder(pixel_valuesixel_values)
        outputs: BaseModelOutputWithNoAttention = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
    
    # def forward(
    #     self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    # ) -> BackboneOutput:
    #     """
    #     Returns:

    #     Examples:

    #     ```python
    #     >>> from transformers import RTDetrResNetConfig, RTDetrResNetBackbone
    #     >>> import torch

    #     >>> config = RTDetrResNetConfig()
    #     >>> model = RTDetrResNetBackbone(config)

    #     >>> pixel_values = torch.randn(1, 3, 224, 224)

    #     >>> with torch.no_grad():
    #     ...     outputs = model(pixel_values)

    #     >>> feature_maps = outputs.feature_maps
    #     >>> list(feature_maps[-1].shape)
    #     [1, 2048, 7, 7]
    #     ```"""
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )

    #     embedding_output = self.embedder(pixel_values)

    #     outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)

    #     hidden_states = outputs.hidden_states

    #     feature_maps = ()
    #     for idx, stage in enumerate(self.stage_names):
    #         if stage in self.out_features:
    #             feature_maps += (hidden_states[idx],)

    #     if not return_dict:
    #         output = (feature_maps,)
    #         if output_hidden_states:
    #             output += (outputs.hidden_states,)
    #         return output

    #     return BackboneOutput(
    #         feature_maps=feature_maps,
    #         hidden_states=outputs.hidden_states if output_hidden_states else None,
    #         attentions=None,
    #     )
        
    # def forward2(self, x: torch.Tensor) -> tuple:
    #     """Forward batch_inputs from the data_preprocessor."""
    #     outs = []
    #     for i, layer_name in enumerate(self.layers):
    #         layer = getattr(self, layer_name)
    #         x = layer(x)
    #         if i in self.out_indices:
    #             outs.append(x)

    #     return tuple(outs)