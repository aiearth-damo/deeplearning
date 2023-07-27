# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader, IterableDataset


if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def _concat_dataset(cfg, default_args=None):
    """Build :obj:`ConcatDataset by."""
    from .dataset_wrappers import ConcatDataset

    img_dir = cfg["img_dir"]
    ann_dir = cfg.get("ann_dir", None)
    split = cfg.get("split", None)
    # pop 'separate_eval' since it is not a valid key for common datasets.
    separate_eval = cfg.pop("separate_eval", True)
    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_img_dir == num_ann_dir or num_ann_dir == 0
        assert num_img_dir == num_split or num_split == 0
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg["img_dir"] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg["ann_dir"] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg["split"] = split[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    """Build datasets."""
    from mmseg.datasets.dataset_wrappers import (
        ConcatDataset,
        MultiImageMixDataset,
        RepeatDataset,
    )

    print(cfg.keys())
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    elif cfg["type"] == "MultiImageMixDataset":
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg["dataset"] = build_dataset(cp_cfg["dataset"])
        cp_cfg.pop("type")
        dataset = MultiImageMixDataset(**cp_cfg)
    elif cfg["type"] == "SemiDataset":
        from aiearth.deeplearning.datasets import SemiDataset

        dataset = SemiDataset(cfg["sup_dataset"], cfg["unsup_dataset"], default_args)
    elif cfg["type"] == "SemiLargeScale":
        from aiearth.deeplearning.datasets import (
            SemiLargeScaleDataset,
        )

        dataset = SemiLargeScaleDataset(
            cfg["sup_dataset"], cfg["unsup_dataset"], default_args
        )
    elif isinstance(cfg.get("img_dir"), (list, tuple)) or isinstance(
        cfg.get("split", None), (list, tuple)
    ):
        dataset = _concat_dataset(cfg, default_args)
    else:
        from mmseg.datasets.builder import DATASETS

        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
