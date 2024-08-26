import torch

def load_original_backbone_weights(url: str):
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')

    state_dict_inner = state_dict['state_dict']
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

    return renamed_state_dict, head_state_dict

def load_original_weights(url: str):
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')

    state_dict_inner = state_dict['state_dict']
    #print(state_dict_inner.keys())

    backbone_state_dict = dict()
    final_dict = dict()
    for k in list(state_dict_inner.keys()):
        if k.startswith('backbone.stem.'):
            backbone_state_dict[k.replace('backbone.stem.', 'embedder.embedder.')] = state_dict_inner[k]
        elif k.startswith('backbone.stage'):
            stripped = k[len('backbone.stage'):]
            stage_num = int(stripped.split('.')[0]) - 1

            old_name = f'backbone.stage{stage_num+1}.'
            new_name = f'encoder.stages.{stage_num}.layers.'

            backbone_state_dict[k.replace(old_name, new_name)] = state_dict_inner[k]
        elif k.startswith('backbone.'):
            backbone_state_dict[k[len('backbone.'):]] = state_dict_inner[k]
        elif k.startswith('neck.'):
            final_dict[k] = state_dict_inner[k]
        elif k.startswith('bbox_head.'):
            final_dict[k] = state_dict_inner[k]
        else:
            #raise ValueError(f"Unexpected key: {k}")
            print('Unexpected key:', k)

    # Merge backbone into final_dict with prefix backbone.model.
    for k, v in backbone_state_dict.items():
        final_dict[f'backbone.model.{k}'] = v

    return final_dict