import torch

import torchvision.transforms as transforms
import requests
from PIL import Image

from transformers import RTMDetCSPNeXtConfig, RTMDetCSPNeXtBackbone
from transformers.models.rtmdet.modeling_rtmdet_cspnext import GlobalAveragePooling, LinearClsHead
from transformers.models.rtmdet.weight_converters import load_original_backbone_weights

backbone_cfgs_by_size = {
    "tiny": RTMDetCSPNeXtConfig(deepen_factor=0.167, widen_factor=0.375),
    "small": RTMDetCSPNeXtConfig(deepen_factor=0.33, widen_factor=0.5),
    "medium": RTMDetCSPNeXtConfig(deepen_factor=0.67, widen_factor=0.75),
}
backbonemodel_name_to_checkpoint_url = {
    "tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth",
    #"tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth",
    "small": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth",
}

def test_backbone_classification(model_name: str = "tiny"):
    renamed_state_dict, head_state_dict = load_original_backbone_weights(backbonemodel_name_to_checkpoint_url[model_name])

    cfg = backbone_cfgs_by_size[model_name]
    model = RTMDetCSPNeXtBackbone(cfg)
    model.load_state_dict(renamed_state_dict)
    model.eval()
    print('Loaded model size:', model_name)

    #img_url = 'https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01496331_electric_ray.JPEG'
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
    output = model.forward(input, return_dict=True)
    neck = GlobalAveragePooling()

    classifier_features = 384 if model_name == "tiny" else 512
    classifier = LinearClsHead(classifier_features, 1000)
    classifier.load_state_dict(head_state_dict)
    x = neck(output.feature_maps)
    #print(x)
    result = classifier(x)
    #print(result.shape)
    # get best class
    class_id = result.argmax(dim=1)
    if class_id != 19:
        raise ValueError(f"Expected class 19, got {class_id}")
    print('Test passed', class_id)

if __name__ == '__main__':
    test_backbone_classification("tiny")
    
