import torch

import torchvision.transforms as transforms
import requests
from PIL import Image

from transformers import RTMDetCSPNeXtConfig, RTMDetCSPNeXtBackbone
from transformers.models.rtmdet.modeling_rtmdet_cspnext import GlobalAveragePooling, LinearClsHead

backbone_cfgs_by_size = {
    "tiny": RTMDetCSPNeXtConfig(deepen_factor=0.167, widen_factor=0.375),
    "small": RTMDetCSPNeXtConfig(deepen_factor=0.33, widen_factor=0.5),
    "medium": RTMDetCSPNeXtConfig(deepen_factor=0.67, widen_factor=0.75),
}
model_name_to_checkpoint_url = {
    #"tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth",
    "tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth",
    "small": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth",
}

def main(model_name: str = "tiny"):
    state_dict = torch.hub.load_state_dict_from_url(model_name_to_checkpoint_url[model_name], map_location='cpu')
    #print(state_dict.keys())
    #['meta', 'state_dict', 'optimizer']

    meta = state_dict['meta']
    # dict_keys(['env_info', 'seed', 'mmcls_version', 'config', 'CLASSES', 'hook_msgs', 'epoch', 'iter', 'mmcv_version', 'time'])
    
    state_dict_inner = state_dict['state_dict']
    # dict_keys(['backbone.stem.0.conv.weight', 'backbone.stem.0.bn.weight', 'backbone.stem.0.bn.bias', 'backbone.stem.0.bn.running_mean', 'backbone.stem.0.bn.running_var', 'backbone.stem.0.bn.num_batches_tracked', 'backbone.stem.1.conv.weight', 'backbone.stem.1.bn.weight', 'backbone.stem.1.bn.bias', 'backbone.stem.1.bn.running_mean', 'backbone.stem.1.bn.running_var', 'backbone.stem.1.bn.num_batches_tracked', 'backbone.stem.2.conv.weight', 'backbone.stem.2.bn.weight', 'backbone.stem.2.bn.bias', 'backbone.stem.2.bn.running_mean', 'backbone.stem.2.bn.running_var', 'backbone.stem.2.bn.num_batches_tracked', 'backbone.stage1.0.conv.weight', 'backbone.stage1.0.bn.weight', 'backbone.stage1.0.bn.bias', 'backbone.stage1.0.bn.running_mean', 'backbone.stage1.0.bn.running_var', 'backbone.stage1.0.bn.num_batches_tracked', 'backbone.stage1.1.main_conv.conv.weight', 'backbone.stage1.1.main_conv.bn.weight', 'backbone.stage1.1.main_conv.bn.bias', 'backbone.stage1.1.main_conv.bn.running_mean', 'backbone.stage1.1.main_conv.bn.running_var', 'backbone.stage1.1.main_conv.bn.num_batches_tracked', 'backbone.stage1.1.short_conv.conv.weight', 'backbone.stage1.1.short_conv.bn.weight', 'backbone.stage1.1.short_conv.bn.bias', 'backbone.stage1.1.short_conv.bn.running_mean', 'backbone.stage1.1.short_conv.bn.running_var', 'backbone.stage1.1.short_conv.bn.num_batches_tracked', 'backbone.stage1.1.final_conv.conv.weight', 'backbone.stage1.1.final_conv.bn.weight', 'backbone.stage1.1.final_conv.bn.bias', 'backbone.stage1.1.final_conv.bn.running_mean', 'backbone.stage1.1.final_conv.bn.running_var', 'backbone.stage1.1.final_conv.bn.num_batches_tracked', 'backbone.stage1.1.blocks.0.conv1.conv.weight', 'backbone.stage1.1.blocks.0.conv1.bn.weight', 'backbone.stage1.1.blocks.0.conv1.bn.bias', 'backbone.stage1.1.blocks.0.conv1.bn.running_mean', 'backbone.stage1.1.blocks.0.conv1.bn.running_var', 'backbone.stage1.1.blocks.0.conv1.bn.num_batches_tracked', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.conv.weight', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.weight', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.bias', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.running_mean', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.running_var', 'backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.num_batches_tracked', 'backbone.stage1.1.blocks.0.conv2.pointwise_conv.conv.weight', 'backbone.stage1.1.blocks.0.conv2.pointwise_conv.bn.weight', 'backbone.stage1.1.blocks.0.conv2.pointwise_conv.bn.bias', 'backbone.stage1.1.blocks.0.conv2.pointwise_conv.bn.running_mean', 'backbone.stage1.1.blocks.0.conv2.pointwise_conv.bn.running_var', 'backbone.stage1.1.blocks.0.conv2.pointwise_conv.bn.num_batches_tracked', 'backbone.stage1.1.attention.fc.weight', 'backbone.stage1.1.attention.fc.bias', 'backbone.stage2.0.conv.weight', 'backbone.stage2.0.bn.weight', 'backbone.stage2.0.bn.bias', 'backbone.stage2.0.bn.running_mean', 'backbone.stage2.0.bn.running_var', 'backbone.stage2.0.bn.num_batches_tracked', 'backbone.stage2.1.main_conv.conv.weight', 'backbone.stage2.1.main_conv.bn.weight', 'backbone.stage2.1.main_conv.bn.bias', 'backbone.stage2.1.main_conv.bn.running_mean', 'backbone.stage2.1.main_conv.bn.running_var', 'backbone.stage2.1.main_conv.bn.num_batches_tracked', 'backbone.stage2.1.short_conv.conv.weight', 'backbone.stage2.1.short_conv.bn.weight', 'backbone.stage2.1.short_conv.bn.bias', 'backbone.stage2.1.short_conv.bn.running_mean', 'backbone.stage2.1.short_conv.bn.running_var', 'backbone.stage2.1.short_conv.bn.num_batches_tracked', 'backbone.stage2.1.final_conv.conv.weight', 'backbone.stage2.1.final_conv.bn.weight', 'backbone.stage2.1.final_conv.bn.bias', 'backbone.stage2.1.final_conv.bn.running_mean', 'backbone.stage2.1.final_conv.bn.running_var', 'backbone.stage2.1.final_conv.bn.num_batches_tracked', 'backbone.stage2.1.blocks.0.conv1.conv.weight', 'backbone.stage2.1.blocks.0.conv1.bn.weight', 'backbone.stage2.1.blocks.0.conv1.bn.bias', 'backbone.stage2.1.blocks.0.conv1.bn.running_mean', 'backbone.stage2.1.blocks.0.conv1.bn.running_var', 'backbone.stage2.1.blocks.0.conv1.bn.num_batches_tracked', 'backbone.stage2.1.blocks.0.conv2.depthwise_conv.conv.weight', 'backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.weight', 'backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.bias', 'backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.running_mean', 'backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.running_var', 'backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.num_batches_tracked', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.conv.weight', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.bn.weight', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.bn.bias', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.bn.running_mean', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.bn.running_var', 'backbone.stage2.1.blocks.0.conv2.pointwise_conv.bn.num_batches_tracked', 'backbone.stage2.1.attention.fc.weight', 'backbone.stage2.1.attention.fc.bias', 'backbone.stage3.0.conv.weight', 'backbone.stage3.0.bn.weight', 'backbone.stage3.0.bn.bias', 'backbone.stage3.0.bn.running_mean', 'backbone.stage3.0.bn.running_var', 'backbone.stage3.0.bn.num_batches_tracked', 'backbone.stage3.1.main_conv.conv.weight', 'backbone.stage3.1.main_conv.bn.weight', 'backbone.stage3.1.main_conv.bn.bias', 'backbone.stage3.1.main_conv.bn.running_mean', 'backbone.stage3.1.main_conv.bn.running_var', 'backbone.stage3.1.main_conv.bn.num_batches_tracked', 'backbone.stage3.1.short_conv.conv.weight', 'backbone.stage3.1.short_conv.bn.weight', 'backbone.stage3.1.short_conv.bn.bias', 'backbone.stage3.1.short_conv.bn.running_mean', 'backbone.stage3.1.short_conv.bn.running_var', 'backbone.stage3.1.short_conv.bn.num_batches_tracked', 'backbone.stage3.1.final_conv.conv.weight', 'backbone.stage3.1.final_conv.bn.weight', 'backbone.stage3.1.final_conv.bn.bias', 'backbone.stage3.1.final_conv.bn.running_mean', 'backbone.stage3.1.final_conv.bn.running_var', 'backbone.stage3.1.final_conv.bn.num_batches_tracked', 'backbone.stage3.1.blocks.0.conv1.conv.weight', 'backbone.stage3.1.blocks.0.conv1.bn.weight', 'backbone.stage3.1.blocks.0.conv1.bn.bias', 'backbone.stage3.1.blocks.0.conv1.bn.running_mean', 'backbone.stage3.1.blocks.0.conv1.bn.running_var', 'backbone.stage3.1.blocks.0.conv1.bn.num_batches_tracked', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.conv.weight', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.weight', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.bias', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.running_mean', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.running_var', 'backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.num_batches_tracked', 'backbone.stage3.1.blocks.0.conv2.pointwise_conv.conv.weight', 'backbone.stage3.1.blocks.0.conv2.pointwise_conv.bn.weight', 'backbone.stage3.1.blocks.0.conv2.pointwise_conv.bn.bias', 'backbone.stage3.1.blocks.0.conv2.pointwise_conv.bn.running_mean', 'backbone.stage3.1.blocks.0.conv2.pointwise_conv.bn.running_var', 'backbone.stage3.1.blocks.0.conv2.pointwise_conv.bn.num_batches_tracked', 'backbone.stage3.1.attention.fc.weight', 'backbone.stage3.1.attention.fc.bias', 'backbone.stage4.0.conv.weight', 'backbone.stage4.0.bn.weight', 'backbone.stage4.0.bn.bias', 'backbone.stage4.0.bn.running_mean', 'backbone.stage4.0.bn.running_var', 'backbone.stage4.0.bn.num_batches_tracked', 'backbone.stage4.1.conv1.conv.weight', 'backbone.stage4.1.conv1.bn.weight', 'backbone.stage4.1.conv1.bn.bias', 'backbone.stage4.1.conv1.bn.running_mean', 'backbone.stage4.1.conv1.bn.running_var', 'backbone.stage4.1.conv1.bn.num_batches_tracked', 'backbone.stage4.1.conv2.conv.weight', 'backbone.stage4.1.conv2.bn.weight', 'backbone.stage4.1.conv2.bn.bias', 'backbone.stage4.1.conv2.bn.running_mean', 'backbone.stage4.1.conv2.bn.running_var', 'backbone.stage4.1.conv2.bn.num_batches_tracked', 'backbone.stage4.2.main_conv.conv.weight', 'backbone.stage4.2.main_conv.bn.weight', 'backbone.stage4.2.main_conv.bn.bias', 'backbone.stage4.2.main_conv.bn.running_mean', 'backbone.stage4.2.main_conv.bn.running_var', 'backbone.stage4.2.main_conv.bn.num_batches_tracked', 'backbone.stage4.2.short_conv.conv.weight', 'backbone.stage4.2.short_conv.bn.weight', 'backbone.stage4.2.short_conv.bn.bias', 'backbone.stage4.2.short_conv.bn.running_mean', 'backbone.stage4.2.short_conv.bn.running_var', 'backbone.stage4.2.short_conv.bn.num_batches_tracked', 'backbone.stage4.2.final_conv.conv.weight', 'backbone.stage4.2.final_conv.bn.weight', 'backbone.stage4.2.final_conv.bn.bias', 'backbone.stage4.2.final_conv.bn.running_mean', 'backbone.stage4.2.final_conv.bn.running_var', 'backbone.stage4.2.final_conv.bn.num_batches_tracked', 'backbone.stage4.2.blocks.0.conv1.conv.weight', 'backbone.stage4.2.blocks.0.conv1.bn.weight', 'backbone.stage4.2.blocks.0.conv1.bn.bias', 'backbone.stage4.2.blocks.0.conv1.bn.running_mean', 'backbone.stage4.2.blocks.0.conv1.bn.running_var', 'backbone.stage4.2.blocks.0.conv1.bn.num_batches_tracked', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.conv.weight', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.weight', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.bias', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.running_mean', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.running_var', 'backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.num_batches_tracked', 'backbone.stage4.2.blocks.0.conv2.pointwise_conv.conv.weight', 'backbone.stage4.2.blocks.0.conv2.pointwise_conv.bn.weight', 'backbone.stage4.2.blocks.0.conv2.pointwise_conv.bn.bias', 'backbone.stage4.2.blocks.0.conv2.pointwise_conv.bn.running_mean', 'backbone.stage4.2.blocks.0.conv2.pointwise_conv.bn.running_var', 'backbone.stage4.2.blocks.0.conv2.pointwise_conv.bn.num_batches_tracked', 'backbone.stage4.2.attention.fc.weight', 'backbone.stage4.2.attention.fc.bias', 'head.fc.weight', 'head.fc.bias'])
    
    # dict_keys(['state', 'param_groups'])

    # remove prefix from keys
    renamed_state_dict = dict()
    head_state_dict = dict()
    for k in list(state_dict_inner.keys()):
        if k.startswith('backbone.stem.'):
            renamed_state_dict[k.replace('backbone.stem.', 'embedder.embedder.')] = state_dict_inner[k]
        elif k.startswith('backbone.stage'):
            stripped = k[len('backbone.stage'):]
            stage_num = int(stripped.split('.')[0]) - 1

            old_name = f'backbone.stage{stage_num+1}.'
            new_name = f'encoder.stages.{stage_num}.layers.'

            renamed_state_dict[k.replace(old_name, new_name)] = state_dict_inner[k]
        elif k.startswith('backbone.'):
            renamed_state_dict[k[len('backbone.'):]] = state_dict_inner[k]
        elif k.startswith('head.'):
            # skip head
            head_state_dict[k[len('head.'):]] = state_dict_inner[k]
        else:
            #raise ValueError(f"Unexpected key: {k}")
            print('Unexpected key:', k)

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
    #print(output.feature_maps)
    #model.save_pretrained(f"rtmdet-cspnext-{model_name}")

    neck = GlobalAveragePooling()

    classifier = LinearClsHead(384, 1000)
    classifier.load_state_dict(head_state_dict)

    x = neck(output.feature_maps)

    #print(x)
    result = classifier(x)
    #print(result.shape)

    # get best class
    print(result.argmax(dim=1))

if __name__ == '__main__':
    main("tiny")
    #main("small")
    
