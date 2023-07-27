# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import numpy as np
import torch.nn.functional as F


class CostomAdaptiveAvgPool2D(nn.Module):
    def __init__(self, output_size, input_size):
        super(CostomAdaptiveAvgPool2D, self).__init__()

        self.output_size = output_size
        self.input_size = input_size

    def forward(self, x):
        H_in, W_in = self.input_size
        H_out, W_out = (
            [self.output_size, self.output_size]
            if isinstance(self.output_size, int)
            else self.output_size
        )

        out_i = []
        for i in range(H_out):
            out_j = []
            for j in range(W_out):
                hs = int(np.floor(i * H_in / H_out))
                he = int(np.ceil((i + 1) * H_in / H_out))

                ws = int(np.floor(j * W_in / W_out))
                we = int(np.ceil((j + 1) * W_in / W_out))

                # print(hs, he, ws, we)
                kernel_size = [he - hs, we - ws]

                out = F.avg_pool2d(x[:, :, hs:he, ws:we], kernel_size)
                out_j.append(out)

            out_j = torch.cat(out_j, -1)
            out_i.append(out_j)

        out_i = torch.cat(out_i, -2)
        return out_i


class CostomPPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(
        self,
        pool_scales,
        in_channels,
        channels,
        conv_cfg,
        norm_cfg,
        act_cfg,
        align_corners,
        export_onnx=False,
        **kwargs
    ):
        super(CostomPPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if export_onnx:
            for pool_scale in pool_scales:
                self.append(
                    nn.Sequential(
                        # nn.AdaptiveAvgPool2d(pool_scale),
                        CostomAdaptiveAvgPool2D(pool_scale, (28, 28)),
                        ConvModule(
                            self.in_channels,
                            self.channels,
                            1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            **kwargs
                        ),
                    )
                )
        else:
            for pool_scale in pool_scales:
                self.append(
                    nn.Sequential(
                        # nn.AdaptiveAvgPool2d(pool_scale),
                        CostomAdaptiveAvgPool2D(pool_scale, (28, 28)),
                        ConvModule(
                            self.in_channels,
                            self.channels,
                            1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            **kwargs
                        ),
                    )
                )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class CostomPSPHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), export_onnx=False, **kwargs):
        super(CostomPSPHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = CostomPPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
            export_onnx=export_onnx,
        )
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
