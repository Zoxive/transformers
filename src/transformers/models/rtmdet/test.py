import torch
from transformers import RTMDetCSPNeXtConfig, RTMDetCSPNeXtBackbone


checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
prefix = 'backbone.'

checkpoint2 = 'https://download.openmmlab.com/mmrazor/v1/rtmdet_distillation/kd_tiny_rtmdet_s_neck_300e_coco/kd_tiny_rtmdet_s_neck_300e_coco_20230213_104240-e1e4197c.pth'

backbone_cfg = RTMDetCSPNeXtConfig()
empty_backbone = RTMDetCSPNeXtBackbone(backbone_cfg)

# load the pretrained weights
state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location='cpu')

empty_backbone.load_state_dict({prefix + k: v for k, v in state_dict.items() if prefix + k in empty_backbone.state_dict()})