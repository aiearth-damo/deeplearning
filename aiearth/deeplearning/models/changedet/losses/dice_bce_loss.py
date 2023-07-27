# -*- conding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class DiceBceLoss(nn.Module):
    def __init__(
        self,
        batch=True,
        reduction="mean",
        loss_name="loss_CEDice",
        loss_weight=1.0,
        focal=False,
    ):
        """
        Args:
            batch (bool): Whether to calculate loss batch-wise.
            reduction (str): The method that reduces the loss to a scalar.
            loss_name (str): The name of the loss.
            loss_weight (float): The weight of the loss.
            focal (bool): Whether to use focal loss.
        """
        super(DiceBceLoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.focal = focal
        self.loss_name = loss_name

    def soft_dice_coeff(self, y_true, y_pred, valid_mask):
        """
        Calculate the soft dice coefficient.

        Args:
            y_true (torch.Tensor): The ground truth tensor.
            y_pred (torch.Tensor): The predicted tensor.
            valid_mask (torch.Tensor): The valid mask tensor.

        Returns:
            torch.Tensor: The soft dice coefficient.
        """
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true * valid_mask)
            j = torch.sum(y_pred * valid_mask)
            intersection = torch.sum(y_true * y_pred * valid_mask)
        else:
            i = (y_true * valid_mask).sum(1).sum(1).sum(1)
            j = (y_pred * valid_mask).sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred * valid_mask).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def focal_loss(self, y_true, y_pred, gamma=2.0, eps=1e-6):
        """
        Calculate the focal loss.

        Args:
            y_true (torch.Tensor): The ground truth tensor.
            y_pred (torch.Tensor): The predicted tensor.
            gamma (float): The gamma value.
            eps (float): The epsilon value.

        Returns:
            torch.Tensor: The focal loss.
        """
        probs = torch.clamp(y_pred, eps, 1.0 - eps)
        targets = torch.clamp(y_true, eps, 1.0 - eps)
        pt = (1 - targets) * (1 - probs) + targets * probs
        loss_focal = -((1.0 - pt) ** gamma) * torch.log(pt)
        # loss_focal = loss_focal.mean()
        return loss_focal

    def soft_dice_loss(self, y_true, y_pred, valid_mask):
        """
        Calculate the soft dice loss.

        Args:
            y_true (torch.Tensor): The ground truth tensor.
            y_pred (torch.Tensor): The predicted tensor.
            valid_mask (torch.Tensor): The valid mask tensor.

        Returns:
            torch.Tensor: The soft dice loss.
        """
        loss = 1 - self.soft_dice_coeff(y_true, y_pred, valid_mask)
        return loss

    def bce_dice(
        self,
        y_pred,
        y_true,
        weight=None,
        reduction="mean",
        ignore_index=255,
        avg_factor=None,
        **kwargs
    ):
        """
        Calculate the BCE dice loss.

        Args:
            y_pred (torch.Tensor): The predicted tensor.
            y_true (torch.Tensor): The ground truth tensor.
            weight (torch.Tensor): The weight tensor.
            reduction (str): The method that reduces the loss to a scalar.
            ignore_index (int): The index to ignore.
            avg_factor (float): The average factor.
            **kwargs: Other arguments.

        Returns:
            torch.Tensor: The BCE dice loss.
        """
        valid_mask = (y_true != ignore_index).float()
        if weight is not None:
            valid_mask *= weight
        y_true = y_true.unsqueeze(1).float()
        y_true[y_true > 0] = 1
        # a =  self.bce_loss(y_pred, y_true)
        if self.focal:
            a = self.focal_loss(y_true, y_pred)
        else:
            a = F.binary_cross_entropy(y_pred, y_true, weight=None, reduction="none")
        a = a.squeeze(1)
        b = self.soft_dice_loss(y_true, y_pred, valid_mask)
        loss = a + b
        loss = weight_reduce_loss(
            loss, weight=valid_mask, reduction=reduction, avg_factor=avg_factor
        )
        return loss

    # def forward(self, score, target, **kwargs):
    #     score = torch.sigmoid(score)
    #     return self.loss_weight * self.bce_dice(score, target)

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        """
        Forward computation.

        Args:
            cls_score (torch.Tensor): The predicted tensor.
            label (torch.Tensor): The ground truth tensor.
            weight (torch.Tensor): The weight tensor.
            avg_factor (float): The average factor.
            reduction_override (str): The method that reduces the loss to a scalar.
            **kwargs: Other arguments.

        Returns:
            torch.Tensor: The loss.
        """
        score = torch.sigmoid(cls_score)
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        # if self.class_weight is not None:
        #     class_weight = cls_score.new_tensor(self.class_weight)
        # else:
        #     class_weight = None
        return self.loss_weight * self.bce_dice(
            score, label, weight, reduction, **kwargs
        )
