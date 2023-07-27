# -*- conding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .efficientnet_pytorch import EfficientNet as EffNet
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint

nonlinearity = partial(F.relu, inplace=True)


@BACKBONES.register_module()
class EfficientNet(nn.Module):
    def __init__(self, config, momentum=0.1, **kwargs):
        super(EfficientNet, self).__init__()
        self.config = config
        if self.config == "effb0":
            model = EffNet.from_pretrained("efficientnet-b0", momentum)
        elif self.config == "effb1":
            model = EffNet.from_pretrained("efficientnet-b1", momentum)
        elif self.config == "effb2":
            model = EffNet.from_pretrained("efficientnet-b2", momentum)
        elif self.config == "effb3":
            model = EffNet.from_pretrained("efficientnet-b3", momentum)
        elif self.config == "effb4":
            model = EffNet.from_pretrained("efficientnet-b4", momentum)
        elif self.config == "effb5":
            model = EffNet.from_pretrained("efficientnet-b5", momentum)
        elif self.config == "effb6":
            model = EffNet.from_name("efficientnet-b6", momentum)
        elif self.config == "effb7":
            model = EffNet.from_name("efficientnet-b7", momentum)
        else:
            raise Exception("Architecture undefined!")
        model.set_swish(memory_efficient=False)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        feature_maps = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(x)
        return feature_maps

    def extract_features_pre(self, x):
        return self.forward(x)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(EfficientNet, self).train(mode)


# #         self._freeze_stages()
#         if mode and self.norm_eval:
#             for m in self.modules():
#                 # trick: eval have effect on BatchNorm only
#                 if isinstance(m, _BatchNorm):
#                     m.eval()
