# -*- conding: utf-8 -*-
import random
import copy
import mmcv
import inspect
import numpy as np
import albumentations

from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.compose import Compose
from mmcv.utils import build_from_cfg
from torchvision import transforms as _transforms


@PIPELINES.register_module()
class StrongWeakAug(object):
    # add by gongyuan
    def __init__(self, pre_transforms, weak_transforms, strong_transforms):
        self.pre_transforms = Compose(pre_transforms)
        self.weak_transforms = Compose(weak_transforms)
        self.strong_transforms = Compose(strong_transforms)

    def __call__(self, results, **kwargs):
        tmp = self.pre_transforms(results)
        tmp2 = {**tmp}
        weak = self.weak_transforms(tmp)
        strong = self.strong_transforms(tmp2)
        return {"weak": weak, "strong": strong}


@PIPELINES.register_module()
class SetIgnoreSeg(object):
    # add by gongyuan
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    def __call__(self, results):
        img = results["img"]
        results["gt_semantic_seg"] = np.zeros(img.shape[:2]) + self.ignore_index
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"set_ignore_index={self.ignore_index}, "
        return repr_str


@PIPELINES.register_module()
class Albu:
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self, transforms):
        self.transforms = copy.deepcopy(transforms)
        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms]
        )

    def albu_builder(self, cfg):
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError("albumentations is not installed")
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
            ]

        return obj_cls(**args)

    def __call__(self, results):
        if "gt_semantic_seg" in results:
            tmp = self.aug(image=results["img"], mask=results["gt_semantic_seg"])
            results["img"] = tmp["image"]
            results["gt_semantic_seg"] = tmp["mask"]
        else:
            tmp = self.aug(image=results["img"])
            results["img"] = tmp["image"]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str


@PIPELINES.register_module()
class RandomAppliedTrans(object):
    """Randomly applied transformations.
    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)
        self.prob = p

    def __call__(self, results):
        return self.trans(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"prob = {self.prob}"
        return repr_str


@PIPELINES.register_module()
class GenerateCutBox:
    def __init__(self, prop_range, n_boxes, crop_size, nomask=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.crop_size = crop_size
        self.nomask = nomask

    def generate_params(self):
        # Choose the proportion of each mask that should be above the threshold
        mask_props = np.random.uniform(
            self.prop_range[0], self.prop_range[1], size=(self.n_boxes)
        )
        # Zeros will cause NaNs, so detect and suppres them
        zero_mask = mask_props == 0.0
        y_props = np.exp(
            np.random.uniform(low=0.0, high=1.0, size=(self.n_boxes))
            * np.log(mask_props)
        )
        x_props = mask_props / y_props
        fac = np.sqrt(1.0 / self.n_boxes)
        y_props *= fac
        x_props *= fac
        y_props[zero_mask] = 0
        x_props[zero_mask] = 0
        sizes = np.round(
            np.stack([y_props, x_props], axis=1) * np.array(self.crop_size)[None, :]
        )
        positions = np.round(
            (np.array(self.crop_size) - sizes)
            * np.random.uniform(low=0.0, high=1.0, size=sizes.shape)
        )
        rectangles = np.append(positions, positions + sizes, axis=1)
        masks = np.zeros(self.crop_size)
        for y0, x0, y1, x1 in rectangles:
            masks[int(y0) : int(y1), int(x0) : int(x1)] = (
                1 - masks[int(y0) : int(y1), int(x0) : int(x1)]
            )
        if self.nomask:
            masks = np.ones_like(masks)
        return masks

    def __call__(self, results):
        cutmask = self.generate_params()
        results["cutmask"] = cutmask
        return results


@PIPELINES.register_module()
class SomeOfAugs(Compose):
    """Select N transforms to apply. Selected transforms will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.
    Args:
        transforms (list): list of transformations to compose.
        n (int): number of transforms to apply.
        replace (bool): Whether the sampled transforms are with or without replacement. Default: True.
        p (float): probability of applying selected transform. Default: 1.
    """

    def __init__(
        self, transforms, n, each_prob=None, replace=False, p=1, replay_mode=False
    ):
        super(SomeOfAugs, self).__init__(transforms)
        self.n = n
        self.replace = replace
        assert sum(each_prob) == 1
        self.each_prob = each_prob
        self.replay_mode = replay_mode
        self.p = p

    def __call__(self, data):
        if self.replay_mode:
            for t in self.transforms:
                data = t(data)
                if data is None:
                    return None
            return data

        if self.each_prob and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
            transforms = random_state.choice(
                self.transforms, size=self.n, replace=self.replace, p=self.each_prob
            )
            for t in transforms:
                data = t(data)
                if data is None:
                    return None
            return data
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str


@PIPELINES.register_module()
class SemiLoadAnnotations(object):
    def __init__(
        self,
        reduce_zero_label=False,
        file_client_args=dict(backend="disk"),
        imdecode_backend="pillow",
    ):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        gt_semantic_seg = (
            np.zeros((results["img_shape"][0], results["img_shape"][1]))
            .squeeze()
            .astype(np.uint8)
        )
        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(reduce_zero_label={self.reduce_zero_label},"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
