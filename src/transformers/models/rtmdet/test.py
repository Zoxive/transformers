import torch
from transformers import RTMDetConfig, RTMDetCSPNeXtBackbone, RTMDetModel


checkpoint = 'https://download.openmmlab.com/mmrazor/v1/rtmdet_distillation/kd_tiny_rtmdet_s_neck_300e_coco/kd_tiny_rtmdet_s_neck_300e_coco_20230213_104240-e1e4197c.pth'

cfg = RTMDetConfig()
empty_backbone = RTMDetModel(cfg)
print('Loading checkpoint...')