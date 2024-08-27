import torch
from transformers import RTMDetConfig, RTMDetCSPNeXtBackbone, RTMDetModel, RTMDetCSPNeXtConfig, RTDetrImageProcessor, RTMDetImageProcessor, RTDetrForObjectDetection
from transformers.models.rtmdet.weight_converters import load_original_weights
from PIL import Image
import requests
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# test
import supervision as sv

class Output:
    def __init__(self, logits, pred_boxes):
        self.logits = torch.tensor(logits)
        self.pred_boxes = torch.tensor(pred_boxes)

# teacher
#checkpoint = 'https://download.openmmlab.com/mmrazor/v1/rtmdet_distillation/kd_tiny_rtmdet_s_neck_300e_coco/kd_tiny_rtmdet_s_neck_300e_coco_20230213_104240-e1e4197c.pth'
# yolo version
checkpoint = 'https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth'
# mmdetection version
#checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

cfg = RTMDetConfig(deepen_factor=0.167, widen_factor=0.375, num_labels=80)
model = RTMDetModel(cfg)
print('Loading checkpoint...')

state_dict = load_original_weights(checkpoint)
model.load_state_dict(state_dict)
model.eval()
print('Loaded model')
img_url = 'https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01592084_chickadee.JPEG'
img_url = 'https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01537544_indigo_bunting.JPEG'
img_url = 'https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n04517823_vacuum.JPEG'
#img_url = 'https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n03770679_minivan.JPEG'
# load the image into memory
img = Image.open(requests.get(img_url, stream=True).raw)
transforms_imagenet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_processor = RTMDetImageProcessor(size={"height": 640, "width": 640})
#image_processor2 = RTDetrImageProcessor(size={"height": 640, "width": 640})

input = image_processor(images=img, return_tensors="pt")
#logits, pred_boxes, outputs = model.forward(**input)
with torch.no_grad():
    outputs = model(**input)

model2 = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
model2.eval()

with torch.no_grad():
    outputs2 = model2(**input)

# num_boxes = len(pred_boxes)

# # for i in range(num_boxes):
# #     box = pred_boxes[i]
# #     logit = logits[i]
# #     best_class = torch.argmax(logit)
# #     confidence = torch.max(logit)
# #     print(box, best_class, confidence)

# for box, logit in zip(pred_boxes, logits):
#     best_class = torch.argmax(logit)
#     confidence = torch.max(logit)
#     if confidence > 0.5:
#         print(box, best_class, confidence)

#output = Output(logits=logits, pred_boxes=pred_boxes)

w,h = img.size
target_size = torch.tensor([[h, w]])
results = image_processor.post_process_object_detection(outputs=outputs, target_sizes=target_size, threshold=0.5)[0]

print('results', len(results))

scores = results['scores']
boxes = results['boxes']
labels = results['labels']

coco_classes = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

# lets draw the boxes on the image

def draw_boxes(img, boxes, labels):
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    for box, logit in zip(boxes, labels):
        best_class = torch.argmax(logit)
        confidence = torch.max(logit)
        if confidence < 0.5:
            continue

        # we need to scale the boxes back to the image size
        
        box = box.detach().numpy()
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(box[0], box[1], best_class, color='red')
    plt.show()

#draw_boxes(img, pred_boxes, logits)
for box, score, label in zip(boxes, scores, labels):
    if (score > 0.6):
        score_val = score.item()
        label_val = label.item()

        class_name = coco_classes[label_val + 1]
        
        print(score_val, box, class_name, label_val)
    #draw_boxes(img, [box], [label])
