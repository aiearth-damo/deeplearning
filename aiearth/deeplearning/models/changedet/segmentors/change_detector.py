# -*- conding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models.builder import SEGMENTORS

# from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from .change_encoder_decoder import ChangeEncoderDecoder
from mmcv.runner import auto_fp16
from mmseg.models import builder


@SEGMENTORS.register_module()
class ChangedetEncoderDecoder(ChangeEncoderDecoder):
    """Encoder Decoder segmentors.

    ChangedetEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(
        self,
        in_one_batch=False,
        concat_channel_neck=False,
        multiclass_auxiliary_head=None,
        edge_auxiliary_head=None,
        danqi_shared_branch=True,
        danqi_concat_feat=True,
        stop_grad_mcd=False,
        stop_grad_cd=False,
        early_fusion=False,
        unet=False,
        **kwargs,
    ):
        """
        Args:
            in_one_batch (bool): Whether to concatenate two images in one batch.
            concat_channel_neck (bool): Whether to concatenate two features in neck.
            multiclass_auxiliary_head (dict): Config dict for multiclass auxiliary head.
            edge_auxiliary_head (dict): Config dict for edge auxiliary head.
            danqi_shared_branch (bool): Whether to share the same head for two branches.
            danqi_concat_feat (bool): Whether to concatenate two features in Danqi's method.
            stop_grad_mcd (bool): Whether to stop gradient propagation in MCD.
            stop_grad_cd (bool): Whether to stop gradient propagation in CD.
            early_fusion (bool): Whether to use early fusion.
            unet (bool): Whether to use UNet.
        """
        super(ChangedetEncoderDecoder, self).__init__(**kwargs)
        self.danqi_shared_branch = danqi_shared_branch
        self.danqi_concat_feat = danqi_concat_feat
        self.stop_grad_mcd = stop_grad_mcd
        self.stop_grad_cd = stop_grad_cd
        self.early_fusion = early_fusion
        self.unet = unet
        self._init_multiclass_auxiliary_head(multiclass_auxiliary_head)
        self._init_edge_auxiliary_head(edge_auxiliary_head)
        self.in_one_batch = in_one_batch
        self.concat_channel_neck = concat_channel_neck
        assert self.with_decode_head

    def _init_multiclass_auxiliary_head(self, multiclass_auxiliary_head):
        """Initialize ``multiclass_auxiliary_head``"""
        self.with_multiclass_auxiliary_head = False
        if multiclass_auxiliary_head is not None:
            self.with_multiclass_auxiliary_head = True
            if not self.danqi_shared_branch:
                self.multiclass_auxiliary_head_1 = builder.build_head(
                    multiclass_auxiliary_head
                )
                self.multiclass_auxiliary_head_2 = builder.build_head(
                    multiclass_auxiliary_head
                )
            else:
                shared_head = builder.build_head(multiclass_auxiliary_head)
                self.multiclass_auxiliary_head_1 = shared_head
                self.multiclass_auxiliary_head_2 = shared_head

    def _init_edge_auxiliary_head(self, edge_auxiliary_head):
        """Initialize ``edge_auxiliary_head``"""
        self.with_edge_auxiliary_head = False
        if edge_auxiliary_head is not None:
            self.with_edge_auxiliary_head = True
            if isinstance(edge_auxiliary_head, list):
                self.edge_auxiliary_head = nn.ModuleList()
                for head_cfg in edge_auxiliary_head:
                    self.edge_auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.edge_auxiliary_head = builder.build_head(edge_auxiliary_head)

    def extract_feat(self, img1, img2):
        """Extract features from images."""
        if self.unet:
            x = self.backbone([img1, img2])
            return x, None, None
        if self.early_fusion:
            merged_img = torch.cat([img1, img2], dim=1)
            x = self.backbone(merged_img)
            return x, None, None

        if self.in_one_batch:
            img = torch.cat([img1, img2], dim=0)
            img = self.backbone(img)
            x1, x2 = [], []
            for ele in img:
                _x1, _x2 = ele.chunk(2)
                x1.append(_x1)
                x2.append(_x2)
        else:
            x1 = self.backbone(img1)
            x2 = self.backbone(img2)
        img = [torch.cat([x1[ind], x2[ind]], dim=1) for ind in range(len(x1))]
        if not self.with_neck:
            return img, x1, x2
        if self.concat_channel_neck:
            x = self.neck(img)
        else:
            x = self.neck([x1, x2])
        return x, x1, x2

    def encode_decode(self, img1, img2, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, x1, x2 = self.extract_feat(img1, img2)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img1.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return out

    def forward_dummy(self, img1, img2):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img1, img2, None)

        return seg_logit

    def _multiclass_auxiliary_head_forward_train(
        self, x1, x2, img_metas, gt_semantic_seg_sat1, gt_semantic_seg_sat2
    ):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        # NOTE: freeze backbones before uperhead
        if self.stop_grad_mcd:
            x1 = [item.detach() for item in x1]
            x2 = [item.detach() for item in x2]
        if self.danqi_concat_feat:
            new_x1 = [
                torch.cat([temp_x1, temp_x2], dim=1) for temp_x1, temp_x2 in zip(x1, x2)
            ]
            new_x2 = [
                torch.cat([temp_x2, temp_x1], dim=1) for temp_x1, temp_x2 in zip(x1, x2)
            ]
            x1 = new_x1
            x2 = new_x2
        losses = dict()
        if isinstance(self.multiclass_auxiliary_head_1, nn.ModuleList):
            for idx, aux_head in enumerate(self.multiclass_auxiliary_head_1):
                loss_aux = aux_head.forward_train(
                    x1, img_metas, gt_semantic_seg_sat1, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, f"multi_aux_1_{idx}"))
            for idx, aux_head in enumerate(self.multiclass_auxiliary_head_2):
                loss_aux = aux_head.forward_train(
                    x2, img_metas, gt_semantic_seg_sat2, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, f"multi_aux_2_{idx}"))
        else:
            loss_aux = self.multiclass_auxiliary_head_1.forward_train(
                x1, img_metas, gt_semantic_seg_sat1, self.train_cfg
            )
            losses.update(add_prefix(loss_aux, "multi_aux_1"))
            loss_aux = self.multiclass_auxiliary_head_2.forward_train(
                x2, img_metas, gt_semantic_seg_sat2, self.train_cfg
            )
            losses.update(add_prefix(loss_aux, "multi_aux_2"))

        return losses

    def _edge_auxiliary_head_forward_train(self, x, img_metas, gt_edges):
        """Run forward function and calculate loss for auxiliary head in
        training."""

        losses = dict()
        if isinstance(self.edge_auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.edge_auxiliary_head):
                loss_aux = aux_head.forward_train(
                    x, img_metas, gt_edges, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, f"edge_aux_{idx}"))
        else:
            loss_aux = self.edge_auxiliary_head.forward_train(
                x, img_metas, gt_edges, self.train_cfg
            )
            losses.update(add_prefix(loss_aux, "edge_aux"))

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
        x, x1, x2 = self.extract_feat(img1, img2)
        if self.stop_grad_cd:
            x = [item.detach() for item in x]

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
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

        if self.with_edge_auxiliary_head:
            loss_aux = self._edge_auxiliary_head_forward_train(
                x, img_metas, kwargs["gt_edges"]
            )
            losses.update(loss_aux)
        return losses

    def forward_test(self, imgs1, imgs2, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """

        if not isinstance(imgs1, list):
            imgs1 = [imgs1]
        if not isinstance(imgs2, list):
            imgs1 = [imgs2]
        if not isinstance(img_metas, list):
            img_metas = [img_metas]

        num_augs = len(imgs1)
        if num_augs != len(img_metas):
            raise ValueError(
                f"num of augmentations ({len(imgs1)}) != "
                f"num of image meta ({len(img_metas)})"
            )
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_["ori_shape"] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_["img_shape"] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_["pad_shape"] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs1[0], imgs2[0], img_metas[0], **kwargs)
        elif self.with_multiclass_auxiliary_head:  # no aug_test for MCD
            return self.simple_test(imgs1[0], imgs2[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs1, imgs2, img_metas, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, img1, img2, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img1, img2, img_metas, **kwargs)
        else:
            return self.forward_test(img1, img2, img_metas, **kwargs)

    # TODO refactor
    def slide_inference(self, img1, img2, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img1.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = torch.zeros(
            [batch_size, num_classes, h_img, w_img], dtype=float, device="cpu"
        )
        count_mat = torch.zeros([batch_size, 1, h_img, w_img], dtype=int, device="cpu")
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img1 = img1[:, :, y1:y2, x1:x2]
                crop_img2 = img2[:, :, y1:y2, x1:x2]
                cur = h_idx * w_grids + w_idx
                total = h_grids * w_grids
                if cur % 1000 == 0:
                    print(f"{cur}/{total}")
                crop_seg_logit = self.encode_decode(crop_img1, crop_img2, img_meta)
                crop_seg_logit = crop_seg_logit.cpu()
                preds[:, :, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0  # cover every patch
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
                device=img1.device
            )
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]["ori_shape"][:2],
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )
        return preds

    def whole_inference(self, img1, img2, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img1, img2, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img1.shape[2:]
            else:
                size = img_meta[0]["ori_shape"][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode="bilinear",
                align_corners=self.align_corners,
                warning=False,
            )

        return seg_logit

    def inference(self, img1, img2, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in [
            "slide",
            "whole",
            "slide_sigmoid",
            "whole_sigmoid",
        ]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == "slide":
            seg_logit = self.slide_inference(img1, img2, img_meta, rescale)
            output = F.softmax(seg_logit, dim=1)
        elif self.test_cfg.mode == "whole":
            seg_logit = self.whole_inference(img1, img2, img_meta, rescale)
            output = F.softmax(seg_logit, dim=1)
        elif self.test_cfg.mode == "slide_sigmoid":
            seg_logit = self.slide_inference(img1, img2, img_meta, rescale)
            output = torch.sigmoid(seg_logit)
            output = output.to(torch.float16)  # save memory
        elif self.test_cfg.mode == "whole_sigmoid":
            seg_logit = self.whole_inference(img1, img2, img_meta, rescale)
            output = torch.sigmoid(seg_logit)
        else:
            ValueError("error test mode {} ".format(self.test_cfg.mode))

        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["original", "horizontal", "vertical", "diagonal"]
            if flip_direction == "horizontal":
                output = output.flip(dims=(3,))
            elif flip_direction == "vertical":
                output = output.flip(dims=(2,))
            elif flip_direction == "diagonal":
                output = output.flip(dims=(2, 3))
            else:
                output = output

        return output

    def simple_test(self, img1, img2, img_meta, rescale=True):
        """Simple test with single image."""
        if torch.onnx.is_in_onnx_export():
            pass
            # img1 = torch.flip(img1, dims=[1])
            # img2 = torch.flip(img2, dims=[1])

        if self.with_multiclass_auxiliary_head:
            return self.simple_test_mcd(img1, img2, img_meta, rescale)
        else:
            return self.simple_test_cd(img1, img2, img_meta, rescale)

    # pylint: disable=no-member, invalid-name
    def simple_test_mcd(self, img1, img2, img_meta, rescale=True):
        """Simple test with single image."""

        assert self.test_cfg.mode == "whole_sigmoid"
        x, x1, x2 = self.extract_feat(img1, img2)
        binary_out = self._decode_head_forward_test(x, img_meta)
        binary_out = resize(
            input=binary_out,
            size=img1.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        binary_out = torch.sigmoid(binary_out)
        thresh = self.test_cfg.thresh
        seg_pred_binary = (binary_out > thresh).long().squeeze(1)
        seg_pred_binary = seg_pred_binary.cpu().to(torch.uint8).numpy()
        fg_prob = binary_out[:, 0, :, :]
        bg_prob = 1.0 - fg_prob
        if len(self.test_cfg.get("semi_probs", [])) > 0:
            threshes = self.test_cfg["semi_probs"]
            semi_pred = [(fg_prob > _).long().squeeze(1) for _ in threshes]
            semi_pred = torch.stack(semi_pred, dim=1)
            seg_pred_raw = (binary_out > thresh).long()
            seg_pred_binary = torch.cat([seg_pred_raw, semi_pred], dim=1)
            seg_pred_binary = seg_pred_binary.cpu().to(torch.uint8).numpy()

        # (N, class_num, 256, 256)
        if self.danqi_concat_feat:
            new_x1 = [
                torch.cat([temp_x1, temp_x2], dim=1) for temp_x1, temp_x2 in zip(x1, x2)
            ]
            new_x2 = [
                torch.cat([temp_x2, temp_x1], dim=1) for temp_x1, temp_x2 in zip(x1, x2)
            ]
            x1 = new_x1
            x2 = new_x2
        results1 = self.multiclass_auxiliary_head_1.forward_test(
            x1, img_meta, self.test_cfg
        )
        results2 = self.multiclass_auxiliary_head_2.forward_test(
            x2, img_meta, self.test_cfg
        )
        results1 = resize(
            input=results1,
            size=img1.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        results2 = resize(
            input=results2,
            size=img1.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if torch.onnx.is_in_onnx_export():
            sat1_pred = results1.argmax(dim=1, keepdim=True)
            sat2_pred = results2.argmax(dim=1, keepdim=True)
            # sat1_pred, sat2_pred in [0, 7]
            # NOTE: 8 is the FG_CLASS_NUM
            fg_class_num = self.multiclass_auxiliary_head_1.num_classes
            if self.test_cfg.mcd_class_name is None:
                combine_pred = sat1_pred * fg_class_num + sat2_pred
            else:
                basic_class = []
                mcd_fg_name = self.test_cfg.mcd_class_name[1:]
                for item in mcd_fg_name:
                    pre_name, post_name = item.split("-")
                    basic_class += [pre_name, post_name]
                basic_class = list(np.unique(basic_class))

                combine_pred = sat1_pred * fg_class_num + sat2_pred

                custom_combine_pred = torch.zeros_like(combine_pred)
                for mcd_index, item in enumerate(mcd_fg_name):
                    pre_name, post_name = item.split("-")
                    pre_index = basic_class.index(pre_name)
                    post_index = basic_class.index(post_name)
                    combined_index = pre_index * fg_class_num + post_index
                    custom_combine_pred[combine_pred == combined_index] = mcd_index
                combine_pred = custom_combine_pred

            # TODO: different architecture results in different output order
            return fg_prob.unsqueeze(1), combine_pred
            # return combine_pred, fg_prob.unsqueeze(1)

        results1 = results1.argmax(dim=1)  # only fg [0, 7]
        results2 = results2.argmax(dim=1)
        results1 = results1.cpu().to(torch.uint8).numpy()
        results2 = results2.cpu().to(torch.uint8).numpy()
        # print(np.unique(results1))
        fg_prob = fg_prob.cpu().numpy()
        bg_prob = bg_prob.cpu().numpy()
        return [
            {
                "seg_pred": seg_pred_binary,
                "fg_prob": fg_prob,
                "bg_prob": bg_prob,
                "results1": results1,
                "results2": results2,
            }
        ]

    def simple_test_cd(self, img1, img2, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img1, img2, img_meta, rescale)
        # return seg_logit
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            return seg_logit

        if "sigmoid" in self.test_cfg.mode:
            thresh = self.test_cfg.thresh
            seg_pred = (seg_logit > thresh).long().squeeze(1)
        else:
            if self.test_cfg.get("high_recall", False):
                seg_logit[:, 0, :, :] *= 0.4
            seg_pred = seg_logit.argmax(dim=1)
        if len(self.test_cfg.get("semi_probs", [])) > 0:
            threshes = self.test_cfg["semi_probs"]
            semi_pred = [(seg_logit > _).long().squeeze(1) for _ in threshes]
            semi_pred = torch.stack(semi_pred, dim=1)
            seg_pred = torch.cat([seg_pred.unsqueeze(1), semi_pred], dim=1)

        seg_pred = seg_pred.cpu().to(torch.uint8).numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def grid_parse(
        self, input_logistic, input_flip_list, input_rotate_list, input_grid_size=3
    ):
        inverse_rotate_list = []
        for cur_rotate in input_rotate_list:
            if cur_rotate == 1:
                inverse_rotate_list.append(1)
            elif cur_rotate == 2:
                inverse_rotate_list.append(4)
            elif cur_rotate == 3:
                inverse_rotate_list.append(3)
            elif cur_rotate == 4:
                inverse_rotate_list.append(2)

        logistic = (
            input_logistic.permute(0, 2, 3, 1)
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        crop_h = logistic.shape[0] // input_grid_size
        crop_w = logistic.shape[1] // input_grid_size

        logistic_pred = None
        for cur_index, cur_flip_mode in enumerate(input_flip_list):
            cur_rotate_mode = inverse_rotate_list[cur_index]
            cur_row_index = cur_index // input_grid_size
            cur_col_index = cur_index % input_grid_size

            cur_logistic = logistic[
                cur_row_index * crop_h : cur_row_index * crop_h + crop_h,
                cur_col_index * crop_w : cur_col_index * crop_w + crop_w,
            ]

            cur_logistic = rotate_image(cur_logistic, cur_rotate_mode)
            cur_logistic = flip_image(cur_logistic, cur_flip_mode)

            if logistic_pred is None:
                logistic_pred = cur_logistic
            else:
                logistic_pred += cur_logistic

        logistic_pred /= len(input_flip_list)
        logistic_pred = np.argmax(logistic_pred, axis=2).astype(np.uint8)
        return logistic_pred

    def aug_test(self, imgs1, imgs2, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs1[0], imgs2[0], img_metas[0], rescale)
        for i in range(1, len(imgs1)):
            cur_seg_logit = self.inference(imgs1[i], imgs2[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs1)

        if "sigmoid" in self.test_cfg.mode:
            thresh = self.test_cfg.thresh
            seg_pred = (seg_logit > thresh).long().squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if len(self.test_cfg.get("semi_probs", [])) > 0:
            threshes = self.test_cfg["semi_probs"]
            semi_pred = [(seg_logit > _).long().squeeze(1) for _ in threshes]
            semi_pred = torch.stack(semi_pred, dim=1)
            seg_pred = torch.cat([seg_pred.unsqueeze(1), semi_pred], dim=1)
        seg_pred = seg_pred.cpu().to(torch.uint8).numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def train_step(self, data_batch, optimizer, **kwargs):
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data_batch["img1"].data)
        )

        return outputs
