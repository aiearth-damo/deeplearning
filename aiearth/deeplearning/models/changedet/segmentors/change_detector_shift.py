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
class ChangedetEncoderDecoderShift(ChangeEncoderDecoder):
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
        shift_loss=False,
        fix_main_branch=False,
        **kwargs,
    ):
        super(ChangedetEncoderDecoderShift, self).__init__(**kwargs)
        self._init_multiclass_auxiliary_head(multiclass_auxiliary_head)
        self._init_edge_auxiliary_head(edge_auxiliary_head)
        self.in_one_batch = in_one_batch
        self.concat_channel_neck = concat_channel_neck
        self.shift_loss = shift_loss
        self.fix_main_branch = fix_main_branch
        assert self.with_decode_head

    def _init_multiclass_auxiliary_head(self, multiclass_auxiliary_head):
        """Initialize ``multiclass_auxiliary_head``"""
        self.with_multiclass_auxiliary_head = False
        if multiclass_auxiliary_head is not None:
            self.with_multiclass_auxiliary_head = True
            if isinstance(multiclass_auxiliary_head, list):
                self.multiclass_auxiliary_head = nn.ModuleList()
                for head_cfg in multiclass_auxiliary_head:
                    self.multiclass_auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.multiclass_auxiliary_head = builder.build_head(
                    multiclass_auxiliary_head
                )

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

    def extract_feat_loss(self, img1, img2, shift):
        """Extract features from images."""
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
            return img, x1, x2, None
        if self.concat_channel_neck:
            x, loss = self.neck.forward_loss(img, shift)
        else:
            x, loss = self.neck.forward_loss([x1, x2], shift)
        return x, x1, x2, loss

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
        losses = dict()
        if self.shift_loss:
            x, x1, x2, loss_encode = self.extract_feat_loss(img1, img2, kwargs["shift"])
            losses.update(loss_encode)
        else:
            x, x1, x2 = self.extract_feat(img1, img2)

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
            # loss_aux = self._edge_auxiliary_head_forward_train(
            #     x, img_metas, gt_semantic_seg)
            loss_aux = self._edge_auxiliary_head_forward_train(
                x, img_metas, kwargs["gt_edges"]
            )
            # loss_aux = self._edge_auxiliary_head_forward_train(
            #     tuple([_.detach() for _ in x]), img_metas, kwargs['gt_edges'])
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
        # preds = img1.new_zeros((batch_size, num_classes, h_img, w_img))
        # count_mat = img1.new_zeros((batch_size, 1, h_img, w_img)).cpu()
        preds = torch.zeros(
            [batch_size, num_classes, h_img, w_img], dtype=float, device="cpu"
        )
        try:
            count_mat = torch.zeros(
                [batch_size, 1, h_img, w_img], dtype=int, device="cpu"
            )
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
                    crop_seg_logit = self.encode_decode(crop_img1, crop_img2, img_meta)
                    # preds += F.pad(crop_seg_logit,
                    #                (int(x1), int(preds.shape[3] - x2), int(y1),
                    #                 int(preds.shape[2] - y2)))
                    crop_seg_logit = crop_seg_logit.cpu()
                    preds[:, :, y1:y2, x1:x2] += crop_seg_logit
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            if torch.onnx.is_in_onnx_export():
                # cast count_mat to constant while exporting to ONNX
                count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
                    device=img1.device
                )
            preds = preds / count_mat
        except:
            pass
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
        # if torch.onnx.is_in_onnx_export():
        #     # our inference backend only support 4D output
        #     seg_pred = seg_pred.unsqueeze(0)
        #     return seg_pred

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

    def train(self, mode=True):
        """Convert the model into training mode will keeping the normalization
        layer freezed."""
        if self.fix_main_branch:
            super(ChangedetEncoderDecoderShift, self).train(False)
            for name, p in self.named_parameters():
                if "neck.flow_make" in name or "neck.attention_op" in name:
                    # if 'decode_head.b1out.' in name or 'decode_head.up.' in name:
                    continue
                p.requires_grad = False
            # for param_tensor in self.state_dict():
            #     print(param_tensor, "\t", self.state_dict()[param_tensor].size())
            self.neck.flow_make[1].train()
            # import pdb; pdb.set_trace()
        else:
            super(ChangedetEncoderDecoderShift, self).train(mode)
