import os
import os.path as osp
import time
import cv2
import numpy as np

import torch
import torch.distributed as dist
import pycocotools.mask as mask_util

import mmcv
from mmcv.image import tensor2imgs
from mmdet.apis.test import collect_results_cpu, collect_results_gpu
from mmdet.apis.train import auto_scale_lr
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_runner,
    get_dist_info,
)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import (
    build_ddp,
    build_dp,
    compat_cfg,
    find_latest_checkpoint,
    get_root_logger,
)


def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).

    Returns:
        list | tuple: RLE encoded mask.
    """
    if isinstance(mask_results, dict):
        mask_results, point_results = mask_results["poly"], mask_results["point"]
    if isinstance(mask_results, tuple):  # mask scoring
        cls_segms, cls_mask_scores = mask_results
    else:
        cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = [[] for _ in range(num_classes)]
    for i in range(len(cls_segms)):
        for cls_segm in cls_segms[i]:
            if len(cls_segm) == 0:
                encoded_mask_results[i].append(None)
                continue
            if cls_segm.shape[-1] == 2:
                # polygon
                msk = np.zeros((300, 300))  # for aicrowd
                # msk = np.zeros((725, 725)) # for inria
                cls_segm = (cls_segm.cpu().numpy() + 0.5).astype(int)
                cv2.drawContours(msk, [cls_segm], -1, 1, -1)
                encoded_mask_results[i].append(
                    mask_util.encode(
                        np.array(msk[:, :, np.newaxis], order="F", dtype="uint8")
                    )[0]
                )  # encoded with RLE
            else:
                encoded_mask_results[i].append(
                    mask_util.encode(
                        np.array(cls_segm[:, :, np.newaxis], order="F", dtype="uint8")
                    )[0]
                )  # encoded with RLE
    if isinstance(mask_results, tuple):
        return encoded_mask_results, cls_mask_scores
    else:
        return encoded_mask_results


def single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, "PALETTE", None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data["img"][0], torch.Tensor):
                img_tensor = data["img"][0]
            else:
                img_tensor = data["img"][0].data[0]
            img_metas = data["img_metas"][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta["img_shape"]
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta["ori_shape"][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta["ori_filename"])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr,
                )

        # encode mask results
        if isinstance(result[0], tuple):
            result = [
                (bbox_results, encode_mask_results(mask_results))
                for bbox_results, mask_results in result
            ]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and "ins_results" in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]["ins_results"]
                result[j]["ins_results"] = (
                    bbox_results,
                    encode_mask_results(mask_results),
                )

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [
                    (bbox_results, encode_mask_results(mask_results))
                    for bbox_results, mask_results in result
                ]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and "ins_results" in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]["ins_results"]
                    result[j]["ins_results"] = (
                        bbox_results,
                        encode_mask_results(mask_results),
                    )

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


from mmdet.core.evaluation import DistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class PolyDistEvalHook(DistEvalHook):
    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(PolyDistEvalHook, self).__init__(*args, **kwargs)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        results = multi_gpu_test(
            runner.model, self.dataloader, tmpdir=tmpdir, gpu_collect=self.gpu_collect
        )
        self.latest_results = results
        if runner.rank == 0:
            print("\n")
            runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)


def train_detector(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = "EpochBasedRunner" if "runner" not in cfg else cfg.runner["type"]

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False,
    )

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get("train_dataloader", {}),
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is None and cfg.get("device", None) == "npu":
        fp16_cfg = dict(loss_scale="dynamic")
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
        custom_hooks_config=cfg.get("custom_hooks", None),
    )

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False,
        )

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get("val_dataloader", {}),
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args["samples_per_gpu"] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = PolyDistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")

    resume_from = None
    if cfg.resume_from is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
