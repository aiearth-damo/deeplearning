# -*- conding: utf-8 -*-
import torch
import torch.nn as nn

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models.builder import SEGMENTORS

# from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from .change_detector import ChangedetEncoderDecoder


@SEGMENTORS.register_module()
class ChangedetSymmetryEncoderDecoder(ChangedetEncoderDecoder):
    def __init__(self, **kwargs):
        super(ChangedetSymmetryEncoderDecoder, self).__init__(**kwargs)

    def extract_feat(self, img1, img2):
        """Extract features from images."""
        x1 = self.backbone(img1)
        x2 = self.backbone(img2)
        out1 = self.neck([x1, x2])
        out2 = self.neck([x2, x1])
        return out1, out2, x1, x2

    def encode_decode(self, img1, img2, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        out1, out2, x1, x2 = self.extract_feat(img1, img2)
        out = self._decode_head_forward_test(out1, img_metas)
        out = resize(
            input=out,
            size=img1.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return out

    def _multiclass_auxiliary_head_forward_train(
        self, x1, x2, img_metas, gt_semantic_seg_sat1, gt_semantic_seg_sat2
    ):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.multiclass_auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.multiclass_auxiliary_head):
                loss_aux = aux_head.forward_train(
                    x1, img_metas, gt_semantic_seg_sat1, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, f"multi_aux_1_{idx}"))
            for idx, aux_head in enumerate(self.multiclass_auxiliary_head):
                loss_aux = aux_head.forward_train(
                    x2, img_metas, gt_semantic_seg_sat2, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, f"multi_aux_2_{idx}"))
        else:
            loss_aux = self.multiclass_auxiliary_head.forward_train(
                x1, img_metas, gt_semantic_seg_sat1, self.train_cfg
            )
            losses.update(add_prefix(loss_aux, "multi_aux_1"))
            loss_aux = self.multiclass_auxiliary_head.forward_train(
                x2, img_metas, gt_semantic_seg_sat2, self.train_cfg
            )
            losses.update(add_prefix(loss_aux, "multi_aux_2"))

        return losses

    def _auxiliary_head_forward_train(self, x1, x2, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(
                    x1, img_metas, gt_semantic_seg, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, f"aux_1_{idx}"))
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(
                    x2, img_metas, gt_semantic_seg, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, f"aux_2_{idx}"))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x1, img_metas, gt_semantic_seg, self.train_cfg
            )
            losses.update(add_prefix(loss_aux, "aux_1"))
            loss_aux = self.auxiliary_head.forward_train(
                x2, img_metas, gt_semantic_seg, self.train_cfg
            )
            losses.update(add_prefix(loss_aux, "aux_2"))

        return losses

    def _decode_head_forward_train(self, x1, x2, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            x1, img_metas, gt_semantic_seg, self.train_cfg
        )

        losses.update(add_prefix(loss_decode, "decode_1"))
        loss_decode = self.decode_head.forward_train(
            x2, img_metas, gt_semantic_seg, self.train_cfg
        )

        losses.update(add_prefix(loss_decode, "decode_2"))
        return losses

    def forward_train(self, img1, img2, img_metas, gt_semantic_seg, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        out1, out2, x1, x2 = self.extract_feat(img1, img2)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            out1, out2, img_metas, gt_semantic_seg
        )
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                out1, out2, img_metas, gt_semantic_seg
            )
            losses.update(loss_aux)

        if self.with_multiclass_auxiliary_head:
            loss_aux = self._multiclass_auxiliary_head_forward_train(
                x1,
                x2,
                img_metas,
                kwargs["gt_semantic_seg_sat1"],
                kwargs["gt_semantic_seg_sat2"],
            )
            losses.update(loss_aux)
        return losses
