# -*- conding: utf-8 -*-
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class EncoderDecoderBinary(EncoderDecoder):
    def __init__(self, **kwargs):
        super(EncoderDecoderBinary, self).__init__(**kwargs)

    def inference(self, img, img_meta, rescale):
        """
        Inference function for EncoderDecoderBinary.

        Args:
            img (Tensor): Input image tensor.
            img_meta (dict): Image meta info.
            rescale (bool): Whether to rescale image.

        Returns:
            Tensor: Output segmentation map.
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
            seg_logit = self.slide_inference(img, img_meta, rescale)
            output = F.softmax(seg_logit, dim=1)
        elif self.test_cfg.mode == "whole":
            seg_logit = self.whole_inference(img, img_meta, rescale)
            output = F.softmax(seg_logit, dim=1)
        elif self.test_cfg.mode == "slide_sigmoid":
            seg_logit = self.slide_inference(img, img_meta, rescale)
            output = torch.sigmoid(seg_logit)
        elif self.test_cfg.mode == "whole_sigmoid":
            seg_logit = self.whole_inference(img, img_meta, rescale)
            output = torch.sigmoid(seg_logit)
        else:
            NotImplementedError()

        flip = img_meta[0]["flip"]
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical", "diagonal"]
            if flip_direction == "horizontal":
                output = output.flip(dims=(3,))
            elif flip_direction == "vertical":
                output = output.flip(dims=(2,))
            elif flip_direction == "diagonal":
                output = output.flip(dims=(2, 3))

        if (
            "rotate90" in img_meta[0]
            and img_meta[0]["rotate90"]
            and img_meta[0]["rotate_degree"] != 0
        ):
            rotate_degree = img_meta[0]["rotate_degree"]
            assert rotate_degree in [0, 90, 180, 270]
            k = int((360 - rotate_degree) / 90)
            output = output.rot90(k, dims=(2, 3))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """
        Simple test function for EncoderDecoderBinary with single image.

        Args:
            img (Tensor): Input image tensor.
            img_meta (dict): Image meta info.
            rescale (bool): Whether to rescale image.

        Returns:
            list: Output segmentation map.
        """
        seg_logit = self.inference(img, img_meta, rescale)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            return seg_logit
        semi_probs = self.test_cfg.get("semi_probs", None)
        if (
            self.test_cfg.mode == "slide_sigmoid"
            or self.test_cfg.mode == "whole_sigmoid"
        ):
            if semi_probs is not None:
                assert len(semi_probs) == 2
                assert semi_probs[0] < semi_probs[1]
                seg_pred = torch.zeros_like(
                    seg_logit, dtype=torch.uint8
                ) + self.test_cfg.get("ignore_index", 255)
                seg_pred[seg_logit < semi_probs[0]] = 0
                seg_pred[seg_logit > semi_probs[1]] = 1
                seg_pred = seg_pred.to(torch.uint8).squeeze(dim=1)
            else:
                seg_pred = (
                    (seg_logit > self.test_cfg.thresh).to(torch.uint8).squeeze(dim=1)
                )
        else:
            assert semi_probs is None
            seg_pred = seg_logit.argmax(dim=1).to(torch.uint8)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """
        Test function with augmentations for EncoderDecoderBinary.
        Only rescale=True is supported.

        Args:
            imgs (list[Tensor]): Input image tensor list.
            img_metas (list[dict]): Image meta info list.
            rescale (bool): Whether to rescale image.

        Returns:
            list: Output segmentation map.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        semi_probs = self.test_cfg.get("semi_probs", None)
        if (
            self.test_cfg.mode == "slide_sigmoid"
            or self.test_cfg.mode == "whole_sigmoid"
        ):
            if semi_probs is not None:
                assert len(semi_probs) == 2
                assert semi_probs[0] <= semi_probs[1]
                seg_pred = torch.zeros_like(
                    seg_logit, dtype=torch.uint8
                ) + self.test_cfg.get("ignore_index", 255)
                seg_pred[seg_logit < semi_probs[0]] = 0
                seg_pred[seg_logit >= semi_probs[1]] = 1
                seg_pred = seg_pred.to(torch.uint8).squeeze(dim=1)
            else:
                seg_pred = (
                    (seg_logit > self.test_cfg.thresh).to(torch.uint8).squeeze(dim=1)
                )
        else:
            assert semi_probs is None
            seg_pred = seg_logit.argmax(dim=1).to(torch.uint8)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    @staticmethod
    def _parse_losses(losses):
        """
        Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for key, value in loss_value.items():
                    log_vars["{}.{}".format(loss_name, key)] = value.mean()
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (
                f"rank {dist.get_rank()}"
                + f" len(log_vars): {len(log_vars)}"
                + " keys: "
                + ",".join(log_vars.keys())
                + "\n"
            )
            assert log_var_length == len(log_vars) * dist.get_world_size(), (
                "loss log variables are different across GPUs!\n" + message
            )

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
