# Copyright (c) OpenMMLab. All rights reserved.
# -*- conding: utf-8 -*-
import torch
import torch.nn as nn
import math

from mmseg.ops import resize
from mmseg.models.losses.cross_entropy_loss import (
    CrossEntropyLoss,
)

from ..builder import DISTILLERSLOSSES


@DISTILLERSLOSSES.register_module()
class ExpSemiLossCPSFAWS7(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        loss_weight2=1.0,
        avg_non_ignore=True,
        ignore_index=255,
        total_iteration=None,
        align_corners=False,
        branch1=True,
        branch2=True,
        teacher1=True,
        teacher2=True,
        end_ratio=2.0,
        scale_factor=None,
        ratio_type="cosine",
        thresh=0.95,
    ):
        super(ExpSemiLossCPSFAWS7, self).__init__()
        self.loss_weight = loss_weight
        self.loss_weight2 = loss_weight2
        self.ignore_index = ignore_index
        self.criterion = CrossEntropyLoss()
        self.end_ratio = end_ratio
        self.total_iteration = total_iteration
        self.align_corners = align_corners
        self.branch1 = branch1
        self.branch2 = branch2
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.scale_factor = scale_factor
        self.ratio_type = ratio_type
        self.thresh = thresh

    def forward(
        self,
        branch1_logits,
        branch1_features,
        conv_seg1,
        estimator_branch1,
        branch2_logits,
        branch2_features,
        conv_seg2,
        estimator_branch2,
        teacher1_logits,
        teacher1_features,
        teacher_conv_seg1,
        estimator_teacher1,
        teacher2_logits,
        teacher2_features,
        teacher_conv_seg2,
        estimator_teacher2,
        current_iter,
        gt_semantic_seg,
    ):
        if self.ratio_type == "cosine":
            ratio = (
                self.end_ratio
                - self.end_ratio
                * (math.cos(math.pi * current_iter / float(self.total_iteration)) + 1)
                / 2
            )
        elif self.ratio_type == "line":
            ratio = self.end_ratio * (current_iter / float(self.total_iteration))
        elif self.ratio_type == None:
            ratio = 0.0
        else:
            NotImplementedError("not implemented")
        """Forward function."""
        if self.scale_factor is not None:
            (
                branch1_logits,
                branch1_features,
                branch2_logits,
                branch2_features,
                teacher1_logits,
                teacher1_features,
                teacher2_logits,
                teacher2_features,
            ) = [
                resize(
                    x,
                    scale_factor=self.scale_factor,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in [
                    branch1_logits,
                    branch1_features,
                    branch2_logits,
                    branch2_features,
                    teacher1_logits,
                    teacher1_features,
                    teacher2_logits,
                    teacher2_features,
                ]
            ]
        sup_batch = gt_semantic_seg.shape[0]
        sup1_logits, sup1_feature, unsup1_logits, unsup1_feature = (
            branch1_logits[:sup_batch],
            branch1_features[:sup_batch],
            branch1_logits[sup_batch:],
            branch1_features[sup_batch:],
        )
        sup2_logits, sup2_feature, unsup2_logits, unsup2_feature = (
            branch2_logits[:sup_batch],
            branch2_features[:sup_batch],
            branch2_logits[sup_batch:],
            branch2_features[sup_batch:],
        )
        sup1_t_logits, sup1_t_feature, unsup1_t_logits, unsup1_t_feature = (
            teacher1_logits[:sup_batch],
            teacher1_features[:sup_batch],
            teacher1_logits[sup_batch:],
            teacher1_features[sup_batch:],
        )
        sup2_t_logits, sup2_t_feature, unsup2_t_logits, unsup2_t_feature = (
            teacher2_logits[:sup_batch],
            teacher2_features[:sup_batch],
            teacher2_logits[sup_batch:],
            teacher2_features[sup_batch:],
        )
        # pdb.set_trace()
        tmp_gt_branch1 = unsup1_logits.argmax(dim=1).unsqueeze(dim=1)
        tmp_gt_branch2 = unsup2_logits.argmax(dim=1).unsqueeze(dim=1)
        tmp_gt_teacher1 = unsup1_t_logits.argmax(dim=1).unsqueeze(dim=1)
        tmp_gt_teacher2 = unsup2_t_logits.argmax(dim=1).unsqueeze(dim=1)
        pro_targets_u1 = (
            torch.cat([sup1_t_logits, unsup1_t_logits], dim=0)
            .softmax(dim=1)
            .max(dim=1)[0]
        )
        pro_targets_u2 = (
            torch.cat([sup2_t_logits, unsup2_t_logits], dim=0)
            .softmax(dim=1)
            .max(dim=1)[0]
        )

        if self.branch1:
            sup1_logits = self._semantic_feature(
                sup1_logits,
                sup1_feature,
                conv_seg1,
                gt_semantic_seg,
                estimator_branch1,
                ratio,
            )
            unsup1_logits = self._semantic_feature(
                unsup1_logits,
                unsup1_feature,
                conv_seg1,
                tmp_gt_branch1,
                estimator_branch1,
                ratio,
            )
        if self.branch2:
            sup2_logits = self._semantic_feature(
                sup2_logits,
                sup2_feature,
                conv_seg2,
                gt_semantic_seg,
                estimator_branch2,
                ratio,
            )
            unsup2_logits = self._semantic_feature(
                unsup2_logits,
                unsup2_feature,
                conv_seg2,
                tmp_gt_branch2,
                estimator_branch2,
                ratio,
            )
        if self.teacher1:
            sup1_t_logits = self._semantic_feature(
                sup1_t_logits,
                sup1_t_feature,
                teacher_conv_seg1,
                gt_semantic_seg,
                estimator_teacher1,
                ratio,
            )
            unsup1_t_logits = self._semantic_feature(
                unsup1_t_logits,
                unsup1_t_feature,
                teacher_conv_seg1,
                tmp_gt_teacher1,
                estimator_teacher1,
                ratio,
            )
        if self.teacher2:
            sup2_t_logits = self._semantic_feature(
                sup2_t_logits,
                sup2_t_feature,
                teacher_conv_seg2,
                gt_semantic_seg,
                estimator_teacher2,
                ratio,
            )
            unsup2_t_logits = self._semantic_feature(
                unsup2_t_logits,
                unsup2_t_feature,
                teacher_conv_seg2,
                tmp_gt_teacher2,
                estimator_teacher2,
                ratio,
            )
        ### cps loss ###
        logits1 = torch.cat([sup1_logits, unsup1_logits], dim=0)
        logits2 = torch.cat([sup2_logits, unsup2_logits], dim=0)
        targets_u1 = torch.cat([sup1_t_logits, unsup1_t_logits], dim=0).argmax(dim=1)
        targets_u2 = torch.cat([sup2_t_logits, unsup2_t_logits], dim=0).argmax(dim=1)
        targets_u11 = logits1.argmax(dim=1)
        targets_u22 = logits2.argmax(dim=1)

        targets_u1[pro_targets_u1 < self.thresh] = self.ignore_index
        targets_u2[pro_targets_u2 < self.thresh] = self.ignore_index

        loss1 = self.criterion(
            logits1.float(), targets_u2.long().detach(), ignore_index=self.ignore_index
        )
        loss2 = self.criterion(
            logits2.float(), targets_u1.long().detach(), ignore_index=self.ignore_index
        )
        loss11 = self.criterion(
            logits1.float(), targets_u22.long().detach(), ignore_index=self.ignore_index
        )
        loss22 = self.criterion(
            logits2.float(), targets_u11.long().detach(), ignore_index=self.ignore_index
        )
        loss = (loss1 + loss2) * self.loss_weight + (
            loss11 + loss22
        ) * self.loss_weight2
        return loss, ratio * torch.ones(1).cuda()

    def _semantic_feature(
        self, preds, features, conv, gt, estimator, ratio, update=True
    ):
        N, A, H, W = features.shape
        _, C, _, _ = preds.shape
        # pdb.set_trace()
        gt_NHW = (
            resize(gt.detach().float(), size=(H, W), mode="nearest", align_corners=None)
            .long()
            .reshape(N * H * W)
        )
        features_NHWxA = features.permute(0, 2, 3, 1).reshape(N * H * W, A)
        preds_NHWxC = preds.permute(0, 2, 3, 1).reshape(N * H * W, C)
        if update:
            with torch.no_grad():
                estimator.update(features_NHWxA.detach(), gt_NHW)
        sv_NHWxC = self._semantic_vector(
            conv,
            features_NHWxA,
            preds_NHWxC,
            gt_NHW,
            estimator.CoVariance.detach(),
            ratio,
        )
        sv = sv_NHWxC.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return sv

    def _semantic_vector(self, conv, features, preds, gt, CoVariance, ratio):
        gt_mask = (gt == self.ignore_index).long()
        labels = (1 - gt_mask).mul(gt).long()

        N = features.size(0)
        A = features.size(1)
        C = preds.shape[1]

        weight_m = list(conv.parameters())[0].squeeze().detach()
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.reshape(N, 1, 1).expand(N, C, A))

        CV_temp = CoVariance[labels]
        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.reshape(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = preds + 0.5 * sigma2.mul(
            (1 - gt_mask).reshape(N, 1).expand(N, C).float()
        )

        return aug_result
