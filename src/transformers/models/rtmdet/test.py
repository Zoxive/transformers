import torch
from transformers import RTMDetConfig, RTMDetCSPNeXtBackbone, RTMDetModel, RTMDetCSPNeXtConfig
from transformers.models.rtmdet.weight_converters import load_original_weights
from PIL import Image
import requests
import torchvision.transforms as transforms

# teacher
#checkpoint = 'https://download.openmmlab.com/mmrazor/v1/rtmdet_distillation/kd_tiny_rtmdet_s_neck_300e_coco/kd_tiny_rtmdet_s_neck_300e_coco_20230213_104240-e1e4197c.pth'
checkpoint = 'https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth'

cfg = RTMDetConfig(deepen_factor=0.167, widen_factor=0.375, num_labels=80)
model = RTMDetModel(cfg)
print('Loading checkpoint...')

state_dict = load_original_weights(checkpoint)
model.load_state_dict(state_dict)
model.eval()
print('Loaded model')
img_url = 'https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01592084_chickadee.JPEG'
# load the image into memory
img = Image.open(requests.get(img_url, stream=True).raw)
transforms_imagenet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input = transforms_imagenet(img).unsqueeze(0)
logits, pred_boxes = model.forward(input)

print('logits', len(logits))
print(logits[0].shape)
print(logits[1].shape)
print(logits[2].shape)
print('pred_boxes', len(pred_boxes))
print(pred_boxes[0].shape)
print(pred_boxes[1].shape)
print(pred_boxes[2].shape)