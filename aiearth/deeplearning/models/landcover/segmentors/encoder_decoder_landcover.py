# -*- conding: utf-8 -*-
import torch
import torch.distributed as dist
from collections import OrderedDict

from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import (
    EncoderDecoder,
)


@SEGMENTORS.register_module()
class EncoderDecoderLandcover(EncoderDecoder):
    def __init__(self, **kwargs):
        super(EncoderDecoderLandcover, self).__init__(**kwargs)

    def simple_test(self, img, img_meta, rescale=True):
        """Perform a simple test with a single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if torch.onnx.is_in_onnx_export():
            return seg_logit
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contains
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

        # If the loss_vars have different lengths, raise an assertion error
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
