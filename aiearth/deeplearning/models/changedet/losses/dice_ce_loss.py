# -*- conding: utf-8 -*-
import torch
import torch.nn as nn

from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weight_reduce_loss

# pylint:disable=no-member, super-with-arguments, too-many-locals


@LOSSES.register_module()
class DiceCeLoss(nn.Module):
    def __init__(self, batch=True, reduction="mean", loss_weight=1.0, ignore_index=255):
        """
        Args:
            batch (bool): Whether to average the loss over the batch_size.
            reduction (str): The method that reduces the loss to a scalar.
            loss_weight (float): The weight of the loss.
            ignore_index (int): The index that is ignored in the loss calculation.
        """
        super(DiceCeLoss, self).__init__()
        self.batch = batch
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.softmax = nn.Softmax(dim=1)

    def soft_dice_coeff(self, y_true, y_pred, valid_mask):
        """
        Calculate the soft dice coefficient.

        Args:
            y_true (torch.Tensor): The ground truth tensor.
            y_pred (torch.Tensor): The predicted tensor.
            valid_mask (torch.Tensor): The mask tensor.

        Returns:
            torch.Tensor: The soft dice coefficient.
        """
        y_probs = self.softmax(y_pred)
        num_classes = y_probs.shape[1]
        nBatch, height, width = y_true.shape
        y_true_nobg = y_true.clone()
        y_true_nobg[y_true_nobg == 255] = 0
        y_true_one_hot = (
            torch.cuda.FloatTensor(num_classes, nBatch, height, width)
            .zero_()
            .scatter_(0, torch.unsqueeze(y_true_nobg, 0), 1)
        )
        y_probs_permute = y_probs.permute(1, 0, 2, 3)

        smooth = 0.0  # may change
        eps = torch.finfo(torch.float32).eps
        if self.batch:
            i = (y_true_one_hot * valid_mask).sum((1, 2, 3))
            j = (y_probs_permute * valid_mask).sum((1, 2, 3))
            intersection = (y_true_one_hot * y_probs_permute * valid_mask).sum(
                (1, 2, 3)
            )
        else:
            i = (y_true_one_hot * valid_mask).sum((2, 3))
            j = (y_probs_permute * valid_mask).sum((2, 3))
            intersection = (y_true_one_hot * y_probs_permute * valid_mask).sum((2, 3))
        score = (2.0 * intersection + smooth) / (i + j + smooth + eps)
        score = score.mean()
        return score

    def soft_dice_loss(self, y_true, y_pred, valid_mask):
        """
        Calculate the soft dice loss.

        Args:
            y_true (torch.Tensor): The ground truth tensor.
            y_pred (torch.Tensor): The predicted tensor.
            valid_mask (torch.Tensor): The mask tensor.

        Returns:
            torch.Tensor: The soft dice loss.
        """
        loss = 1 - self.soft_dice_coeff(y_true, y_pred, valid_mask)
        return loss

    def ce_dice(
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
        Calculate the cross entropy loss and the soft dice loss.

        Args:
            y_pred (torch.Tensor): The predicted tensor.
            y_true (torch.Tensor): The ground truth tensor.
            weight (torch.Tensor): The weight tensor.
            reduction (str): The method that reduces the loss to a scalar.
            ignore_index (int): The index that is ignored in the loss calculation.
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple[torch.Tensor]: The cross entropy loss and the soft dice loss.
        """

        # y_true with 255 as ignore
        valid_mask = (y_true != ignore_index).float()
        if not weight is None:
            valid_mask *= weight
        # NOTE: this will be masked in the end, very careful!!!
        dice_loss = self.soft_dice_loss(y_true, y_pred, valid_mask)
        y_true = y_true.long()
        ce_loss = self.ce_loss(y_pred, y_true)
        return ce_loss, dice_loss

    def forward(
        self,
        cls_logit,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        """
        Forward computation.

        Args:
            cls_logit (torch.Tensor): The predicted tensor.
            label (torch.Tensor): The ground truth tensor.
            weight (torch.Tensor): The weight tensor.
            avg_factor (int): Average factor that is used to average the loss.
            reduction_override (str): The method that reduces the loss to a scalar.

        Returns:
            torch.Tensor: The loss.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        return self.ce_dice(cls_logit, label, weight, reduction, **kwargs)
