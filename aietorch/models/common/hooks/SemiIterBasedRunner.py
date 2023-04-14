# -*- conding: utf-8 -*-
from mmcv.runner.iter_based_runner import IterBasedRunner
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
import torch


@RUNNERS.register_module()
class SemiIterBasedRunner(IterBasedRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook("before_train_iter")
        if "semi" in data_batch:
            img1 = data_batch["origin"]["img"].data
            gt_semantic_seg1 = data_batch["origin"]["gt_semantic_seg"].data
            img_metas1 = data_batch["origin"]["img_metas"].data
            img2 = data_batch["semi"]["img"].data
            gt_semantic_seg2 = data_batch["semi"]["gt_semantic_seg"].data
            img_metas2 = data_batch["semi"]["img_metas"].data
            data_batch = {}
            data_batch["img_metas"] = img_metas1 + img_metas2
            data_batch["img"] = torch.cat(img1 + img2, dim=0)
            data_batch["gt_semantic_seg"] = torch.cat(
                gt_semantic_seg1 + gt_semantic_seg2, dim=0
            )
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError("model.train_step() must return a dict")
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.outputs = outputs
        self.call_hook("after_train_iter")
        self._inner_iter += 1
        self._iter += 1


@RUNNERS.register_module()
class SemiEpochBasedRunner(EpochBasedRunner):
    def run_iter(self, data_batch, train_mode, **kwargs):
        if "semi" in data_batch:
            img1 = data_batch["origin"]["img"].data
            gt_semantic_seg1 = data_batch["origin"]["gt_semantic_seg"].data
            img_metas1 = data_batch["origin"]["img_metas"].data
            img2 = data_batch["semi"]["img"].data
            gt_semantic_seg2 = data_batch["semi"]["gt_semantic_seg"].data
            img_metas2 = data_batch["semi"]["img_metas"].data
            data_batch = {}
            data_batch["img_metas"] = img_metas1 + img_metas2
            data_batch["img"] = torch.cat(img1 + img2, dim=0)
            data_batch["gt_semantic_seg"] = torch.cat(
                gt_semantic_seg1 + gt_semantic_seg2, dim=0
            )
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs
            )
        elif train_mode:
            outputs = self.model.train_step(
                data_batch, self.optimizer, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError(
                '"batch_processor()" or "model.train_step()"'
                'and "model.val_step()" must return a dict'
            )
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.outputs = outputs
