import torch
from transformers.models.rtmdet.configuration_rtmdet import RTMDetCSPNeXtConfig
from transformers.models.rtmdet.modeling_rtmdet_cspnext import RTMDetCSPNeXtBackbone


checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
prefix = 'backbone.'

backbone_cfg = RTMDetCSPNeXtConfig()
empty_backbone = RTMDetCSPNeXtBackbone(backbone_cfg)

# load the pretrained weights
state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location='cpu')

empty_backbone.load_state_dict({prefix + k: v for k, v in state_dict.items() if prefix + k in empty_backbone.state_dict()})