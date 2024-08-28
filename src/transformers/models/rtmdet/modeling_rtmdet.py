from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from transformers.models.rt_detr.modeling_rt_detr import RTDetrObjectDetectionOutput
from .CSPNeXtPAFPN import CSPNeXtPAFPN
from .RTMDetHead import RTMDetHead
from ...utils.generic import ModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_rtmdet import RTMDetConfig
from torch import Tensor
import torch.nn as nn
from .modeling_rtmdet_cspnext import RTMDetCSPNeXtBackbone

@dataclass
class RTMDetObjectDetectionOutput(ModelOutput):
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None

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
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        outputs = self.backbone(pixel_values)

        sequence_output = outputs[0]

        # logits = self.class_labels_classifier(sequence_output)
        # pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        
        features = self.neck(sequence_output)
        output = self.bbox_head(features)

        img_width, img_height = pixel_values.shape[-2:]
        logits, pred_boxes = self.bbox_head.pred_test(*output, img_width=img_width, img_height=img_height)

        loss, loss_dict, auxiliary_outputs = None, None, None

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + (auxiliary_outputs,) + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output
        
        # TODO own class
        return RTDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
        )
