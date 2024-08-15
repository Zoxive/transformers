# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""RTMDET configuration"""

import math
from typing import Sequence

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin

logger = logging.get_logger(__name__)


class RTMDetConfig(PretrainedConfig):
    model_type = "rtmdet"

    def __init__(
        self,
        # backbone
        backbone_config=None,
        **kwargs,
    ):
        if backbone_config is None:
            backbone_config = RTMDetCSPNeXtConfig()
        
        self.backbone_config = backbone_config
        super().__init__(**kwargs)

class RTMDetCSPNeXtConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = 'rtmdet_cspnext'
    
    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        #use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        #conv_cfg = None,
        #norm_cfg = dict(type='BN', momentum=0.03, eps=0.001),
        #act_cfg = dict(type='SiLU'),
        norm_eval: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.arch = arch
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        #self.use_depthwise = use_depthwise
        self.expand_ratio = expand_ratio
        self.arch_ovewrite = arch_ovewrite
        self.spp_kernel_sizes = spp_kernel_sizes
        self.channel_attention = channel_attention
        #self.conv_cfg = conv_cfg
        #self.norm_cfg = norm_cfg
        #self.act_cfg = act_cfg
        self.norm_eval = norm_eval
