# Copyright (c) OpenMMLab. All rights reserved.
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
from mmdet.core import (
    multi_apply,
    bbox_xyxy_to_cxcywh,
    bbox_cxcywh_to_xyxy,
    reduce_mean,
)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.dense_heads.detr_head import DETRHead


@HEADS.register_module()
class DeformablePolyHead(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        num_points=36,
        loss_poly=None,
        loss_poly_score=None,
        poly_mode="normal",
        with_points=False,
        loss_point=None,
        warp_points=False,
        key_points=True,
        refine_num=0,
        image_size=320,
        **kwargs,
    ):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_points = num_points
        self.poly_mode = poly_mode
        self.with_points = with_points
        self.warp_points = warp_points
        self.key_points = key_points
        self.refine_num = refine_num
        self.image_size = image_size
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage

        super(DeformablePolyHead, self).__init__(
            *args, transformer=transformer, **kwargs
        )
        self.loss_poly = build_loss(loss_poly)
        self.loss_poly_score = build_loss(loss_poly_score)
        if self.with_points:
            self.point_convs = nn.ModuleList()
            for i in range(2):
                self.point_convs.append(
                    ConvModule(
                        256,
                        256,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type="GN", num_groups=32),
                    )
                )
            self.point_convs.append(
                ConvModule(
                    256,
                    1,
                    1,
                    1,
                    act_cfg=None,
                )
            )
            bias_init = bias_init_with_prob(0.01)
            self.point_convs[-1].conv.bias.data.fill_(bias_init)
            self.loss_point = build_loss(loss_point)

        if self.refine_num > 0:
            # concat
            self.merge_op = ConvModule(256, 256, 1, stride=1, padding=0)
            self.refine_modules = []
            for _idx in range(self.refine_num):
                self.refine_modules.append(Refine(c_in=256, num_point=36, stride=4.0))

        # self.renderer = nr.Renderer(camera_mode='look_at', image_size=self.image_size, light_intensity_ambient=1,
        #                             light_intensity_directional=1, perspective=False)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)
        poly_branch = []
        for _ in range(self.num_reg_fcs):
            poly_branch.append(Linear(self.embed_dims, self.embed_dims))
            poly_branch.append(nn.ReLU())
        poly_branch.append(Linear(self.embed_dims, self.num_points * 2))
        poly_branch = nn.Sequential(*poly_branch)

        poly_cls = []
        for _ in range(self.num_reg_fcs):
            poly_cls.append(Linear(self.embed_dims, self.embed_dims))
            poly_cls.append(nn.ReLU())
        poly_cls.append(Linear(self.embed_dims, self.num_points))
        poly_cls = nn.Sequential(*poly_cls)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
            self.poly_branches = nn.ModuleList([poly_branch for _ in range(num_pred)])
            self.poly_cls_branches = nn.ModuleList([poly_cls for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        # if self.loss_poly_score.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.poly_cls_branches:
        #         nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.poly_branches:
        #     constant_init(m[-1], 0, bias=0)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:])
                .to(torch.bool)
                .squeeze(0)
            )
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord,
        ) = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches
            if self.with_box_refine
            else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_polys = []
        outputs_poly_clses = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            _reference = reference
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp_p = self.poly_branches[lvl](hs[lvl])
            poly_cls = self.poly_cls_branches[lvl](hs[lvl])
            b, q, _ = tmp_p.shape
            tmp_p = tmp_p.reshape((b, q, -1, 2))
            if reference.shape[-1] == 4:
                tmp += reference
                tmp_p += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                if self.poly_mode == "offset":
                    tmp_p[:, :, 1:, :] += tmp_p[:, :, :-1, :]
                    tmp_p[:, :, :1, :] += reference.unsqueeze(2)
                elif self.poly_mode == "normal":
                    tmp_p[..., :2] += reference.unsqueeze(2)
                elif self.poly_mode == "linear_reg":
                    tmp_p = (tmp_p.sigmoid() - 0.5) + _reference.unsqueeze(2)
                else:
                    assert False
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.poly_mode == "linear_reg":
                outputs_poly = tmp_p
            else:
                outputs_poly = tmp_p.sigmoid()
            outputs_polys.append(outputs_poly)
            outputs_poly_cls = poly_cls
            outputs_poly_clses.append(outputs_poly_cls)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_polys = torch.stack(outputs_polys)
        outputs_poly_clses = torch.stack(outputs_poly_clses)
        if self.as_two_stage:
            assert False
            return (
                outputs_classes,
                outputs_coords,
                outputs_polys,
                outputs_poly_clses,
                enc_outputs_class,
                enc_outputs_coord.sigmoid(),
            )
        else:
            if self.with_points:
                _input = mlvl_feats[0]
                for _idx in range(1, 4):
                    _input += F.interpolate(
                        mlvl_feats[_idx], size=(40, 40), mode="bilinear"
                    )
                for m in self.point_convs:
                    _input = m(_input)
                point_pred = _input
            else:
                point_pred = None
            if self.refine_num > 0:
                _input = mlvl_feats[0]
                for _idx in range(1, 4):
                    _input += F.interpolate(
                        mlvl_feats[_idx], size=(40, 40), mode="bilinear"
                    )
                for m in self.point_convs:
                    _input = m(_input)
                _input = self.merge_op(_input)
                outputs_polys = self.refine_modules[0](_input, outputs_polys)

            return (
                outputs_classes,
                outputs_coords,
                outputs_polys,
                outputs_poly_clses,
                None,
                None,
                point_pred,
            )

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        poly_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_poly_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            poly_targets_list,
            poly_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            poly_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_poly_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            poly_targets_list,
            poly_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        poly_pred,
        gt_bboxes,
        gt_labels,
        gt_polys,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        """ "Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore
        )
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta["img_shape"]

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # poly targets
        """
        poly_weights = poly_pred.new_zeros((self.num_query, self.num_points, 2))
        poly_weights[pos_inds, :4] = 1.0
        """
        poly_targets = []
        poly_weights = []
        # import pdb
        # pdb.set_trace()
        index = assign_result.gt_inds[assign_result.gt_inds > 0].cpu().numpy()
        for idx in range(self.num_query):
            if assign_result.gt_inds[idx] > 0:
                poly_targets.append(
                    poly_pred.new_tensor(
                        gt_polys.masks[assign_result.gt_inds[idx] - 1][0][None]
                    )
                )
                poly_weights.append(
                    poly_pred.new_tensor(
                        gt_polys.poly_weights[assign_result.gt_inds[idx] - 1][0][None]
                    )
                )
            else:
                poly_targets.append(poly_pred.new_zeros((self.num_points, 2))[None])
                poly_weights.append(poly_pred.new_zeros((self.num_points, 1))[None])
        poly_targets = torch.cat(poly_targets, dim=0)
        poly_weights = torch.cat(poly_weights, dim=0)
        factor = poly_pred.new_tensor([img_w, img_h]).unsqueeze(0).unsqueeze(0)
        poly_targets = poly_targets / factor
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            poly_targets,
            poly_weights,
            pos_inds,
            neg_inds,
        )

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        poly_preds,
        poly_scores,
        gt_bboxes_list,
        gt_labels_list,
        gt_polys_list,
        img_metas,
        gt_poly_masks_list=None,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        poly_preds_list = [poly_preds[i] for i in range(num_imgs)]
        poly_scores_list = [poly_scores[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            poly_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_polys_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            poly_targets_list,
            poly_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        poly_targets = torch.cat(poly_targets_list, 0)
        poly_weights = torch.cat(poly_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta["img_shape"]
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )

        # poly L1 loss
        factors = []
        for img_meta, poly_pred in zip(img_metas, poly_preds):
            img_h, img_w, _ = img_meta["img_shape"]
            factor = (
                poly_pred.new_tensor([img_w, img_h])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.num_query, self.num_points, 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)
        poly_preds = poly_preds.reshape(-1, self.num_points, 2)
        # poly_preds = poly_preds * factors
        # poly_targets = poly_targets * factors

        loss_poly = self.loss_poly(
            poly_preds, poly_targets, poly_weights, avg_factor=num_total_pos
        )
        if self.key_points:
            _new_weight = poly_weights.new_zeros(poly_weights.shape)
            _new_weight[(poly_weights.squeeze(-1) == 5)] = 1
            # _new_weight[(poly_weights.sum(axis=1) > 0).reshape(-1)] = 1
            loss_poly_score = self.loss_poly_score(
                poly_scores.reshape(-1),
                _new_weight.reshape(-1),
                avg_factor=cls_avg_factor * self.num_points,
            )
        else:
            loss_poly_score = self.loss_poly_score(
                poly_scores.reshape(-1),
                poly_weights.reshape(-1),
                avg_factor=cls_avg_factor * self.num_points,
            )

        loss_poly_mask = torch.zeros_like(loss_poly_score)

        return (
            loss_cls,
            loss_bbox,
            loss_iou,
            loss_poly,
            loss_poly_score,
            poly_weights,
            loss_poly_mask,
        )

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        all_poly_preds,
        all_poly_scores,
        enc_cls_scores,
        enc_bbox_preds,
        all_point_scores,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_poly_masks_list=None,
        gt_bboxes_ignore=None,
    ):
        """ "Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_polys_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_poly_masks_list = [gt_poly_masks_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        (
            losses_cls,
            losses_bbox,
            losses_iou,
            losses_poly,
            losses_poly_score,
            poly_weights,
            losses_poly_mask,
        ) = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_poly_preds,
            all_poly_scores,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_polys_list,
            img_metas_list,
            all_gt_poly_masks_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i]) for i in range(len(img_metas))
            ]
            (
                enc_loss_cls,
                enc_losses_bbox,
                enc_losses_iou,
                enc_losses_poly,
            ) = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                gt_bboxes_list,
                binary_labels_list,
                img_metas,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_loss_iou"] = enc_losses_iou
            loss_dict["enc_loss_poly"] = enc_losses_poly

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        loss_dict["loss_poly"] = losses_poly[-1]
        loss_dict["loss_poly_score"] = losses_poly_score[-1]
        loss_dict["loss_poly_mask"] = losses_poly_mask[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for (
            loss_cls_i,
            loss_bbox_i,
            loss_iou_i,
            loss_poly_i,
            losses_poly_score_i,
            losses_poly_mask_i,
        ) in zip(
            losses_cls[:-1],
            losses_bbox[:-1],
            losses_iou[:-1],
            losses_poly[:-1],
            losses_poly_score[:-1],
            losses_poly_mask[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_poly"] = loss_poly_i
            loss_dict[f"d{num_dec_layer}.loss_poly_score"] = losses_poly_score_i
            loss_dict[f"d{num_dec_layer}.loss_poly_mask"] = losses_poly_mask_i
            num_dec_layer += 1

        if self.with_points:
            batch_size = all_point_scores.shape[0]
            point_targets = all_point_scores.new_zeros(
                (all_point_scores.shape[0], 1, 320, 320)
            )
            for bs in range(batch_size):
                for _polys, _polys_weight in zip(
                    gt_masks_list[bs].masks, gt_masks_list[bs].poly_weights
                ):
                    for _poly, _poly_weight in zip(_polys[0], _polys_weight[0]):
                        if _poly_weight > 0:
                            point_targets[bs][0] = gen_gaussian_target(
                                point_targets[bs][0], _poly, 2
                            )

            DEBUG = False
            if DEBUG:
                import cv2
                import numpy as np
                import os.path as osp

                msk = F.interpolate(point_targets, 300)
                for bs in range(batch_size):
                    imsk = msk[bs][0].cpu().numpy()
                    img = cv2.imread(img_metas[bs]["filename"])
                    img[imsk > 0.1] = (
                        img[imsk > 0.1] * 0.5 + np.array((0, 255, 255))[None] * 0.5
                    )
                    cv2.imwrite(
                        osp.join("vis_debug", osp.basename(img_metas[bs]["filename"])),
                        img,
                    )

            # point Focal loss
            all_point_scores = F.interpolate(all_point_scores, 320)
            loss_point = self.loss_point(
                all_point_scores.sigmoid(),
                point_targets,
                avg_factor=max(1, point_targets.eq(1).sum()),
            )
            loss_dict["loss_point"] = loss_point
            if self.with_points:
                grid = (all_poly_preds[-1] - 0.5) * 2
                base_feat = all_point_scores.detach()
                sample_points = F.grid_sample(base_feat, grid)
                sample_weights = poly_weights[-1].reshape(
                    -1, 1, self.num_query, self.num_points
                )
                sample_targets = sample_weights.clone()
                loss_warp = F.binary_cross_entropy_with_logits(
                    sample_points,
                    sample_targets,
                    weight=sample_weights,
                    reduction="sum",
                )
                loss_dict["loss_warp"] = loss_warp / sample_weights.sum()

        return loss_dict

    def _get_bboxes_single(
        self,
        cls_score,
        bbox_pred,
        poly_pred,
        poly_score,
        point_score,
        img_shape,
        scale_factor,
        rescale=False,
    ):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get("max_per_img", self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
            poly_pred = poly_pred[bbox_index]
            poly_score = poly_score[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        poly_pred[..., 0] = poly_pred[..., 0] * img_shape[1]
        poly_pred[..., 1] = poly_pred[..., 1] * img_shape[0]
        if rescale:
            poly_pred /= poly_pred.new_tensor(scale_factor[:2])

        if point_score is not None:
            point_score = point_score.sigmoid()
        else:
            point_score = None
        if self.key_points:
            poly_score = poly_score
        return det_bboxes, det_labels, poly_pred, poly_score, point_score

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def get_bboxes(
        self,
        all_cls_scores,
        all_bbox_preds,
        all_poly_preds,
        all_poly_scores,
        enc_cls_scores,
        enc_bbox_preds,
        all_point_scores,
        img_metas,
        rescale=False,
    ):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        poly_preds = all_poly_preds[-1]
        all_poly_scores = all_poly_scores[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            poly_pred = poly_preds[img_id]
            all_poly_score = all_poly_scores[img_id]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            proposals = self._get_bboxes_single(
                cls_score,
                bbox_pred,
                poly_pred,
                all_poly_score,
                all_point_scores,
                img_shape,
                scale_factor,
                rescale,
            )
            result_list.append(proposals)
        return result_list

    def forward_train(
        self,
        x,
        img_metas,
        gt_bboxes,
        gt_labels=None,
        gt_masks=None,
        gt_poly_masks=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        **kwargs,
    ):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (
                gt_bboxes,
                gt_labels,
                gt_masks,
                img_metas,
                gt_poly_masks,
            )
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


class Refine(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, stride=4.0):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.trans_feature = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, 256, kernel_size=3, padding=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.trans_poly = torch.nn.Linear(
            in_features=((num_point + 1) * 64), out_features=num_point * 4, bias=False
        )
        self.trans_fuse = torch.nn.Linear(
            in_features=num_point * 4, out_features=num_point * 2, bias=True
        )

    def global_deform(self, points_features, init_polys):
        poly_num = init_polys.size(0)
        points_features = self.trans_poly(points_features)
        offsets = self.trans_fuse(points_features).view(poly_num, self.num_point, 2)
        coarse_polys = offsets * self.stride + init_polys.detach()
        return coarse_polys

    def forward(self, feature, ct_polys, init_polys, ct_img_idx, ignore=False):
        if ignore or len(init_polys) == 0:
            return init_polys
        h, w = feature.size(2), feature.size(3)
        poly_num = ct_polys.size(0)

        feature = self.trans_feature(feature)

        ct_polys = ct_polys.unsqueeze(1).expand(
            init_polys.size(0), 1, init_polys.size(2)
        )
        points = torch.cat([ct_polys, init_polys], dim=1)
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(
            poly_num, -1
        )
        coarse_polys = self.global_deform(feature_points, init_polys)
        return coarse_polys


def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.0) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.0) - 1
    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros(
        [img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]
    ).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i : i + 1], poly)[
            0
        ].permute(1, 0, 2)
        gcn_feature[ind == i] = feature
    return gcn_feature
