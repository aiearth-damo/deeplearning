# -*- conding: utf-8 -*-
import torch
import torch.nn as nn

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.runner import force_fp32
from mmseg.ops import resize
from mmseg.models.losses import accuracy


@HEADS.register_module()
class BuildingChangeHead(BaseDecodeHead):
    """This class implements the Fully Convolution Networks for Semantic Segmentation.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        num_convs (int): Number of convolutions in the head. Default: 2.
        kernel_size (int): The kernel size for convolutions in the head. Default: 3.
        concat_input (bool): Whether to concatenate the input and output of convolutions
            before the classification layer.
        dilation (int): The dilation rate for convolutions in the head. Default: 1.
        freeze (bool): Whether to freeze the model. Default: False.
        momentum (float): The momentum for the batch normalization layer. Default: 0.01.
    """

    def __init__(
        self,
        in_channels,
        channels,
        num_convs=2,
        kernel_size=3,
        concat_input=True,
        dilation=1,
        freeze=False,
        momentum=0.01,
        **kwargs
    ):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.freeze = freeze
        super(BuildingChangeHead, self).__init__(in_channels, channels, **kwargs)

        self.b1out = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.BatchNorm2d(self.channels, momentum=momentum),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(self.channels, self.num_classes, 2, 2)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        output = self.b1out(inputs)
        output = self.up2(output)
        # output = torch.sigmoid(output)
        if self.freeze:
            output = output.detach()
        return output

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, gt_weights=None):
        """Compute segmentation loss.

        Args:
            seg_logit (torch.Tensor): The segmentation logits.
            seg_label (torch.Tensor): The segmentation labels.
            gt_weights (torch.Tensor): The ground truth weights. Default: None.

        Returns:
            dict: The loss dictionary.
        """
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        elif gt_weights is not None:
            seg_weight = gt_weights
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        """
        print('ignore_index',self.ignore_index)
        print('seg_weight',seg_weight)
        print('seg_label shape',seg_label.shape)
        print('seg_logit shape',seg_logit.shape)
        print('seg_label',seg_label.max())
        print('seg_logit',seg_logit.argmax(dim=1).max())
        """
        loss["loss_seg"] = self.loss_decode(
            seg_logit, seg_label, weight=seg_weight, ignore_index=self.ignore_index
        )
        loss["acc_seg"] = accuracy(seg_logit, seg_label)
        return loss
