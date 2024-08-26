from typing import Dict, List, Optional
import torch
from transformers.models.rtmdet.CSPNeXtPAFPN import CSPNeXtPAFPN
from transformers.models.rtmdet.RTMDetHead import RTMDetHead
from ...modeling_utils import PreTrainedModel
from .configuration_rtmdet import RTMDetConfig
from torch import Tensor
import torch.nn as nn
from .modeling_rtmdet_cspnext import RTMDetCSPNeXtBackbone

class RTMDetPreTrainedModel(PreTrainedModel):
    config_class = RTMDetConfig
    base_model_prefix = "rtmdet"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        #raise NotImplementedError()
        pass

class RTMDetConvEncoder(nn.Module):
    def __init__(self, config: RTMDetConfig):
        super().__init__()

        self.model = RTMDetCSPNeXtBackbone(config.backbone_config)

    def forward(self, pixel_values: Tensor):
        features = self.model(pixel_values)

        # out = []
        # for feature_map in features:
        #     # do we need this mask?
        #     mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
        #     out.append((feature_map, mask))
        #     #out.append(feature_map)

        # return out
        return features

# TODO
size_to_config = {
     "tiny": dict(deepen_factor=0.167, widen_factor=0.375),
     "small": dict(deepen_factor=0.33, widen_factor=0.5),
     "medium": dict(deepen_factor=0.67, widen_factor=0.75),
}

class RTMDetModel(RTMDetPreTrainedModel):
    def __init__(self, config: RTMDetConfig):
        super().__init__(config)

        deepen_factor = config.deepen_factor
        widen_factor = config.widen_factor

        self.backbone = RTMDetConvEncoder(config)
        # TODO fix params
        self.neck = CSPNeXtPAFPN(
            in_channels=[256, 512, 1024],
            out_channels=256,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
        )
        self.bbox_head = RTMDetHead(
            num_classes=config.num_labels, 
            in_channels=256,
            widen_factor=widen_factor,
            feat_channels=256,
        )

        self.init_weights()

    def forward(self, 
        pixel_values: Tensor,
        labels: Optional[List[Dict]] = None,
    ):
        outputs = self.backbone(pixel_values)

        sequence_output = outputs[0]
        
        features = self.neck(sequence_output)
        output = self.bbox_head(features)

        return output