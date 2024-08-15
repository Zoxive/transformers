from ...modeling_utils import PreTrainedModel
from .configuration_rtmdet import RTMDetConfig
import torch.nn as nn
from .modeling_rtmdet_cspnext import RTMDetCSPNeXtBackbone

class RTMDetPreTrainedModel(PreTrainedModel):
    config_class = RTMDetConfig
    base_model_prefix = "rtmdet"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        raise NotImplementedError()

class RTMDetConvEncoder(nn.Module):
    def __init__(self, config: RTMDetConfig):
        super().__init__()

        backbone = RTMDetCSPNeXtBackbone(config)
        
        self.model = backbone

class RTMDetModel(RTMDetPreTrainedModel):

    def __init__(self, config: RTMDetConfig):
        super().__init__(config)

        self.backbone = RTMDetConvEncoder(config)
        self.neck = None
        self.head = None

        self.init_weights()