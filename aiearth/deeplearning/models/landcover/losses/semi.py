# Copyright (c) OpenMMLab. All rights reserved.
# -*- conding: utf-8 -*-
import torch
import torch.nn as nn

from ..builder import DISTILLERSLOSSES
from mmseg.models.losses.cross_entropy_loss import (
    CrossEntropyLoss,
)


@DISTILLERSLOSSES.register_module()
class SemiLossCPS(nn.Module):
    def __init__(self, loss_weight=1.0, avg_non_ignore=True, ignore_index=255):
        super(SemiLossCPS, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.criterion = CrossEntropyLoss(
            loss_weight=loss_weight, avg_non_ignore=avg_non_ignore
        )

    def forward(self, strong_logits, weak_logits):
        max_probs, targets_u = torch.max(weak_logits, dim=1)
        loss1 = self.criterion(
            strong_logits.float(), targets_u.long(), ignore_index=self.ignore_index
        )

        max_probs2, targets_u2 = torch.max(strong_logits, dim=1)
        loss2 = self.criterion(
            weak_logits.float(), targets_u2.long(), ignore_index=self.ignore_index
        )

        loss = loss1 + loss2
        return loss
