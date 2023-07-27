# -*- conding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES, build_loss


@LOSSES.register_module()
class BCELoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        ignore_index=255,
        reduction="none",
        loss_name="loss_bce",
        removeIgnore=False,
        **kwargs
    ):
        """
        BCELoss is a binary cross-entropy loss function.
        """
        super(BCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self._loss_name = loss_name
        self.removeIgnore = removeIgnore

    def forward(self, cls_score, label, **kwargs):
        """
        Forward function of BCELoss.
        """
        if label.dim() == 3:
            label = label.unsqueeze(dim=1)
        valid_mask = label.clone() != self.ignore_index
        label[~valid_mask] = 1
        # cls_score = torch.sigmoid(cls_score)
        loss_cls = self.loss_weight * F.binary_cross_entropy_with_logits(
            cls_score, label.float(), reduction=self.reduction
        )
        if self.removeIgnore:
            loss_cls = loss_cls[valid_mask]
            loss_cls = torch.mean(loss_cls)
        else:
            loss_cls = torch.mean(loss_cls)
        return loss_cls

    @property
    def loss_name(self):
        return self._loss_name


@LOSSES.register_module()
class BinaryDiceLoss(nn.Module):
    def __init__(
        self,
        batch=False,
        loss_weight=1.0,
        ignore_index=255,
        smooth=0.0,
        loss_name="loss_binarydice",
        iou=False,
        converIgnore=False,
        sigmoid=True,
    ):
        """
        BinaryDiceLoss is a binary dice loss function.
        """
        super(BinaryDiceLoss, self).__init__()
        self.batch = batch
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_name = loss_name
        self.smooth = smooth
        self.iou = iou
        self.converIgnore = converIgnore
        self.sigmoid = sigmoid

    def soft_dice_coeff(self, y_true, y_pred):
        """
        Calculate the soft dice coefficient.
        """
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        y_true = torch.unsqueeze(y_true, dim=1)
        if self.converIgnore:
            y_true[y_true == self.ignore_index] = 0
        valid_mask = y_true != self.ignore_index
        BatchSize = valid_mask.shape[0]
        # may change
        if self.batch:
            uniou_i = torch.sum(y_true * valid_mask)
            uniou_j = torch.sum(y_pred * valid_mask)
            intersection = torch.sum(y_true * y_pred * valid_mask)
        else:
            uniou_i = (y_true * valid_mask).reshape(BatchSize, -1).sum(1)
            uniou_j = (y_pred * valid_mask).reshape(BatchSize, -1).sum(1)
            intersection = (y_true * y_pred * valid_mask).reshape(BatchSize, -1).sum(1)
        if self.iou:
            score = (intersection + self.smooth) / (
                uniou_i + uniou_j - intersection + self.smooth
            )  # iou
        else:
            score = (2.0 * intersection + self.smooth) / (
                uniou_i + uniou_j + self.smooth
            )
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        """
        Calculate the soft dice loss.
        """
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def forward(self, cls_score, label, **kwargs):
        """
        Forward function of BinaryDiceLoss.
        """
        return self.loss_weight * self.soft_dice_loss(label, cls_score)


@LOSSES.register_module()
class EnsembleLoss(nn.Module):
    def __init__(self, loss_dict_list=[], loss_name="ensemble", **kwargs):
        """
        EnsembleLoss is a loss function that combines multiple loss functions.
        """
        super(EnsembleLoss, self).__init__()
        self.loss_list = []
        self.loss_name = loss_name
        assert "loss" not in self.loss_name
        for cur_loss_dict in loss_dict_list:
            self.loss_list.append(build_loss(cur_loss_dict))

    def forward(self, input_data, label, **kwargs):
        """
        Forward function of EnsembleLoss.
        """
        losses = {}
        for cur_loss_object in self.loss_list:
            cur_loss = cur_loss_object(input_data.clone(), label.clone(), **kwargs)
            losses["{}".format(cur_loss_object.loss_name)] = cur_loss
        return losses


@LOSSES.register_module()
class IouLoss(nn.Module):
    def __init__(
        self, loss_weight=1.0, ignore_index=255, loss_name="loss_multiIoU", **kwargs
    ):
        """
        IouLoss is a loss function that calculates the intersection over union.
        """
        super(IouLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss_name = loss_name

    def to_one_hot(self, pred, target):
        """
        Convert the target to one-hot encoding.
        """
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None

    def forward(self, input, target, **kwargs):
        """
        Forward function of IouLoss.
            logit => N x Classes x H x W
            target => N x H x W
        """
        class_num = input.size(1)
        N = len(input)
        pred = F.softmax(input, dim=1)
        target_onehot, mask = self.to_one_hot(pred, target)

        # Numerator Product
        inter = pred * target_onehot
        union = pred + target_onehot - inter

        if mask is not None:
            inter[mask] = 0
            union[mask] = 0
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, class_num, -1).sum(2)

        # Denominator
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, class_num, -1).sum(2)

        loss = inter / (union + 1e-5)
        loss = 1 - loss.mean()

        # Return average loss over classes and batch
        return loss * self.loss_weight
