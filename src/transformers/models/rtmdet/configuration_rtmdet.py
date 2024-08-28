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
from typing import Sequence, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

logger = logging.get_logger(__name__)


class RTMDetConfig(PretrainedConfig):
    model_type = "rtmdet"

    def __init__(
        self,
        backbone_config=None,
        num_labels: int = 80,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        **kwargs,
    ):
        if backbone_config is None:
            backbone_config = RTMDetCSPNeXtConfig(deepen_factor=deepen_factor, widen_factor=widen_factor)
        
        self.backbone_config = backbone_config
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        super().__init__(**kwargs)
        self.num_labels = num_labels

class RTMDetCSPNeXtConfig(BackboneConfigMixin, PretrainedConfig):
    model_type = 'rtmdet_cspnext'
    
    def __init__(
        self,
        num_channels: int = 3,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes: Union[int, Sequence[int]] = 5,
        channel_attention: bool = True,
        norm_eval: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.arch = arch
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor

        # TODO can we get the length from the arch_settings instead of a hardcoded value?
        stages = arch == 'P6' and 5 or 4
        
        self.stage_names = ['stem'] + [f'stage{i}' for i in range(1, stages + 1)]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.expand_ratio = expand_ratio
        self.arch_ovewrite = arch_ovewrite
        self.spp_kernel_sizes = spp_kernel_sizes
        self.channel_attention = channel_attention
        self.norm_eval = norm_eval

        # self._out_features, self._out_indices = get_aligned_output_features_output_indices(
        #     out_features=None, out_indices=out_indices, stage_names=self.stage_names
        # )
