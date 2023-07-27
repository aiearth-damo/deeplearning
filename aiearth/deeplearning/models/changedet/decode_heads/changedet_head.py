# -*- conding: utf-8 -*-
import torch
import torch.nn as nn

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from .uper_head import CostomUPerHead
from mmcv.runner import force_fp32
from mmseg.ops import resize


def accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate accuracy according to the prediction and target.
    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.
    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.0) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), f"maxk {maxk} exceeds pred dimension {pred.size(1)}"
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    if ignore_index is not None:
        correct = correct[:, target != ignore_index]
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        # Avoid causing ZeroDivisionError when all pixels
        # of an image are ignored
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = target[target != ignore_index].numel() + eps
        else:
            total_num = target.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res


@HEADS.register_module()
class ChangeDetHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of channels in the head.
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
        freeze (bool): Whether to freeze the parameters of the head. Default: False.
        momentum (float): The momentum of the normalization layer. Default: 0.01.
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
        **kwargs,
    ):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.freeze = freeze
        super(ChangeDetHead, self).__init__(in_channels, channels, **kwargs)

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
        """
        Forward function.

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
        """
        Compute segmentation loss.

        Args:
            seg_logit (torch.Tensor): The predicted segmentation tensor.
            seg_label (torch.Tensor): The ground truth segmentation tensor.
            gt_weights (torch.Tensor): The ground truth weights tensor. Default: None.

        Returns:
            dict: A dictionary containing the loss and accuracy.
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
        loss_decode = self.loss_decode(
            seg_logit, seg_label, weight=seg_weight, ignore_index=self.ignore_index
        )
        if type(self.loss_decode).__name__ == "DiceBceLoss":
            # for cd, [4, 896, 896]
            loss["loss_seg"] = loss_decode
        elif type(self.loss_decode).__name__ == "DiceCeLoss":
            # for mcd, two value tensor
            loss["loss_ce"], loss["loss_dice"] = loss_decode[0], loss_decode[1]
        else:
            loss["loss_seg"] = loss_decode
        # loss['loss_ce'], loss['loss_dice'] = loss_decode[0], loss_decode[1]
        loss["acc_seg"] = accuracy(seg_logit, seg_label)
        return loss


@HEADS.register_module()
class ChangeDetMCDUPerHead(CostomUPerHead):
    """Why not use built-in UPerHead:
    (1) Custom multi-class accuracy without bg implementaed here.
    (2) Built-in mmseg does not support ce loss with ignore_index,
        thus no extra loss file needed.
    """

    def __init__(
        self,
        only_ce_loss=False,
        kernel_upsample=False,
        num_classes=-1,
        mcd_class_name=None,
        **kwargs,
    ):
        if mcd_class_name is not None:
            basic_class = []
            for item in mcd_class_name[1:]:
                pre, post = item.split("-")
                basic_class.append(pre)
                basic_class.append(post)
            num_classes = len(set(basic_class))
        super(ChangeDetMCDUPerHead, self).__init__(
            export_onnx=True, num_classes=num_classes, **kwargs
        )
        self.only_ce_loss = only_ce_loss
        self.kernel_upsample = kernel_upsample
        if kernel_upsample:
            head_channels = 16
            self.danqi_b1out = nn.Sequential(
                nn.Conv2d(self.channels, head_channels, 3, padding=1),
                nn.BatchNorm2d(head_channels, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(head_channels, head_channels, 2, 2),
                nn.BatchNorm2d(head_channels, momentum=0.1),
                nn.ReLU(inplace=True),
            )
            self.danqi_up2 = nn.ConvTranspose2d(head_channels, self.num_classes, 2, 2)

    def _forward_feature(self, inputs):
        """Just copy from latest mmseg code.
        Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        if self.kernel_upsample:
            output = self.danqi_b1out(output)
            output = self.danqi_up2(output)  # (N, class_num, 896, 896)
        else:
            output = self.cls_seg(output)
        return output

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, gt_weights=None):
        """Compute segmentation loss."""
        loss = {}
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
        seg_label_raw = seg_label.clone()
        loss_decode = self.loss_decode(
            seg_logit, seg_label, weight=seg_weight, ignore_index=self.ignore_index
        )
        if self.only_ce_loss:
            loss["MCD_loss_ce"] = loss_decode[0]
        else:
            loss["MCD_loss_ce"], loss["MCD_loss_dice"] = loss_decode[0], loss_decode[1]

        # mask = seg_label != 255    # (N, 896, 896), [0, 8]
        # total_gt_pixel = mask.sum()    # a value, might be zeros
        # pred_class = seg_logit.argmax(dim=1)    # [0, 7] fg class
        # fg_label = seg_label    # [0, 7, 255] fg class
        # tp = fg_label == pred_class    # (N, 896, 896)
        # tp_pixel = tp.sum().float()
        # loss['MCD_acc_seg'] = tp_pixel/(total_gt_pixel+1e-5)
        loss["MCD_acc_seg"] = accuracy(
            seg_logit, seg_label_raw, ignore_index=self.ignore_index
        )
        return loss
