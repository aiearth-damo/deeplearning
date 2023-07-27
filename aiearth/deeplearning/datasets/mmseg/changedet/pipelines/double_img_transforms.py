from mmseg.datasets.pipelines.transforms import *
from mmseg.datasets.builder import PIPELINES
from scipy.ndimage.morphology import distance_transform_edt
import cv2
import numpy as np
from collections import OrderedDict

"""
This file contains the implementation of various data augmentation pipelines for the DoubleImageDataset class. 
These pipelines include:
- DoubleImageResize: Resizes the input images to a given size.
- DoubleImageRandomFlip: Randomly flips the input images horizontally or vertically.
- DoubleImagePad: Pads the input images to a given size.
- DoubleImageNormalize: Normalizes the input images.
- DoubleImageRandomCrop: Randomly crops the input images.
- DoubleImagePhotoMetricDistortion: Applies photometric distortion to the input images.
- DoubleImageLoadAsBinaryLabel: Converts the input segmentation maps to binary labels.
- BinaryLabelDilate: Dilates the binary labels.
- OmmitSmallRegion: Omits small regions from the binary labels.
- DoubleImageRandomRgbBgr: Randomly swaps the color channels of the input images.
...

"""


@PIPELINES.register_module()
class DoubleImageResize(Resize):
    """
    Resize the input images to a given size.
    Required keys are "img1" and "img2", added or modified keys are "img1", "img2", "img_shape", "pad_shape", "scale_factor", "keep_ratio", "scale", and "scale_idx".
    Args:
        img_scale (tuple[int]): (w, h) of resized image. Default: None.
        multiscale_mode (str): Either "range" or "value". Default: "range".
        ratio_range (tuple[float]): (min_ratio, max_ratio) of the resized image. Default: None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the image. Default: True.
    """

    def __init__(self, **kwargs):
        super(DoubleImageResize, self).__init__(**kwargs)

    def _random_scale(self, results):
        """
        Randomly sample a scale from the given scale range or list.
        Args:
            results (dict): A result dict contains the img1 and img2.
        """
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results["img1"].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range
                )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """
        Resize images with ``results['scale']``.
        Args:
            results (dict): A result dict contains the img1 and img2.
        """
        if self.keep_ratio:
            img1, scale_factor = mmcv.imrescale(
                results["img1"], results["scale"], return_scale=True
            )
            img2, scale_factor = mmcv.imrescale(
                results["img2"], results["scale"], return_scale=True
            )
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img1.shape[:2]
            h, w = results["img1"].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img1, w_scale, h_scale = mmcv.imresize(
                results["img1"], results["scale"], return_scale=True
            )
            img2, w_scale, h_scale = mmcv.imresize(
                results["img2"], results["scale"], return_scale=True
            )
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results["img1"] = img1
        results["img2"] = img2
        results["img_shape"] = img1.shape
        results["pad_shape"] = img1.shape  # in case that there is no padding
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = self.keep_ratio


@PIPELINES.register_module()
class DoubleImageRandomFlip(RandomFlip):
    """
    Randomly flip the input images horizontally or vertically.
    Required keys are "img1" and "img2", added or modified keys are "img1", "img2", "flip", and "flip_direction".
    Args:
        prob (float): Probability of flipping. Default: 0.5.
        direction (str): The flipping direction. Options are "horizontal" and "vertical". Default: "horizontal".
    """

    def __init__(self, **kwargs):
        super(DoubleImageRandomFlip, self).__init__(**kwargs)

    def __call__(self, results):
        """
        Call function to flip the input images.
        Args:
            results (dict): A result dict contains the img1 and img2.
        Returns:
            results (dict): A result dict contains the img1, img2, flip, and flip_direction.
        """
        if "flip" not in results:
            flip = True if np.random.rand() < self.prob else False
            results["flip"] = flip
        if "flip_direction" not in results:
            results["flip_direction"] = self.direction
        if results["flip"]:
            # flip image
            results["img1"] = mmcv.imflip(
                results["img1"], direction=results["flip_direction"]
            )
            results["img2"] = mmcv.imflip(
                results["img2"], direction=results["flip_direction"]
            )

            # flip segs
            for key in results.get("seg_fields", []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results["flip_direction"]
                ).copy()
        return results


@PIPELINES.register_module()
class DoubleImagePad(Pad):
    """
    Pad the input images to a given size.
    Required keys are "img1" and "img2", added or modified keys are "img1", "img2", "pad_shape", "pad_fixed_size", and "pad_size_divisor".
    Args:
        size (tuple[int]): Fixed size of padding. Default: None.
        size_divisor (int): The divisor of padded size. Default: None.
        pad_val (float | int | tuple[int]): Padding value. Default: 0.
    """

    def __init__(self, **kwargs):
        super(DoubleImagePad, self).__init__(**kwargs)

    def _pad_img(self, results):
        """
        Pad images according to ``self.size`` or ``self.size_divisor``.
        Args:
            results (dict): A result dict contains the img1 and img2.
        """
        if self.size is not None:
            if (
                results["img1"].shape[0] < self.size[0]
                and results["img1"].shape[1] < self.size[1]
            ):
                padded_img1 = mmcv.impad(
                    results["img1"], shape=self.size, pad_val=self.pad_val
                )
                padded_img2 = mmcv.impad(
                    results["img2"], shape=self.size, pad_val=self.pad_val
                )
            else:
                padded_img1 = results["img1"]
                padded_img2 = results["img2"]
        elif self.size_divisor is not None:
            padded_img1 = mmcv.impad_to_multiple(
                results["img1"], self.size_divisor, pad_val=self.pad_val
            )
            padded_img2 = mmcv.impad_to_multiple(
                results["img2"], self.size_divisor, pad_val=self.pad_val
            )
        results["img1"] = padded_img1
        results["img2"] = padded_img2
        results["pad_shape"] = padded_img1.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor


@PIPELINES.register_module()
class DoubleImageNormalize(Normalize):
    """
    Normalize the input images.
    Required keys are "img1" and "img2", added or modified keys are "img1", "img2", and "img_norm_cfg".
    Args:
        mean (tuple[float]): Mean values of 3 channels.
        std (tuple[float]): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB. Default: True.
    """

    def __init__(self, split=None, **kwargs):
        super(DoubleImageNormalize, self).__init__(**kwargs)
        self.split = split

    def __call__(self, results):
        """
        Call function to normalize the input images.
        Args:
            results (dict): A result dict contains the img1 and img2.
        Returns:
            results (dict): A result dict contains the img1, img2, and img_norm_cfg.
        """
        if self.split is not None:
            img1 = results["img1"]
            assert img1.shape[2] == len(self.split)
            img1 = img1[:, :, self.split]
            results["img1"] = img1

            img2 = results["img2"]
            assert img2.shape[2] == len(self.split)
            img2 = img2[:, :, self.split]
            results["img"] = img2

        results["img1"] = mmcv.imnormalize(
            results["img1"], self.mean, self.std, self.to_rgb
        )
        results["img2"] = mmcv.imnormalize(
            results["img2"], self.mean, self.std, self.to_rgb
        )
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class DoubleImageRandomCrop(RandomCrop):
    """
    Crop the input images and semantic segmentation maps with random size and aspect ratio.
    Required keys are "img1" and "img2", added or modified keys are "img1", "img2", "img_shape", and "seg_fields".
    Args:
        crop_size (tuple[int]): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio between the cropped region and the visible region of a category. Default: 1.0.
        ignore_index (int): The label index to be ignored in semantic segmentation maps. Default: 255.
    """

    def __init__(self, **kwargs):
        super(DoubleImageRandomCrop, self).__init__(**kwargs)

    def __call__(self, results):
        """
        Call function to crop the input images and semantic segmentation maps.
        Args:
            results (dict): A result dict contains the img1 and img2.
        Returns:
            results (dict): A result dict contains the img1, img2, img_shape, and seg_fields.
        """
        img1 = results["img1"]
        img2 = results["img2"]
        crop_bbox = self.get_crop_bbox(img1)
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results["gt_semantic_seg"], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img1)

        # crop the image
        img1 = self.crop(img1, crop_bbox)
        img2 = self.crop(img2, crop_bbox)
        img_shape = img1.shape
        results["img1"] = img1
        results["img2"] = img2
        results["img_shape"] = img_shape

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        return results


@PIPELINES.register_module()
class DoubleImagePhotoMetricDistortion(PhotoMetricDistortion):
    """
    Distort the input images' photo metric.
    Required keys are "img1" and "img2".
    Args:
        brightness_delta (int): Delta of brightness.
        contrast_range (tuple[float]): Range of contrast.
        saturation_range (tuple[float]): Range of saturation.
        hue_delta (int): Delta of hue.
    """

    def __init__(self, **kwargs):
        super(DoubleImagePhotoMetricDistortion, self).__init__(**kwargs)

    def __call__(self, results):
        """
        Call function to distort the input images' photo metric.
        Args:
            results (dict): A result dict contains the img1 and img2.
        Returns:
            results (dict): A result dict contains the img1 and img2.
        """
        for key in ["img1", "img2"]:
            img = results[key]
            # random brightness
            img = self.brightness(img)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                img = self.contrast(img)

            # random saturation
            img = self.saturation(img)

            # random hue
            img = self.hue(img)

            # random contrast
            if mode == 0:
                img = self.contrast(img)

            results[key] = img
        return results


@PIPELINES.register_module()
class DoubleImageLoadAsBinaryLabel(object):
    """
    Load the input images as binary label.
    Required keys are "gt_semantic_seg", added or modified keys are "gt_semantic_seg".
    Args:
        ignore_labels (list[int]): The label index to be ignored in semantic segmentation maps. Default: [].
        ignore_ind (int): The label index to be ignored in binary label. Default: 255.
    """

    def __init__(self, ignore_labels=[], ignore_ind=255):
        if not isinstance(ignore_labels, list):
            ignore_labels = [ignore_labels]
        self.ignore_labels = ignore_labels
        self.ignore_ind = ignore_ind

    def __call__(self, results):
        """
        Call function to load the input images as binary label.
        Args:
            results (dict): A result dict contains the gt_semantic_seg.
        Returns:
            results (dict): A result dict contains the gt_semantic_seg.
        """
        seg_map = results["gt_semantic_seg"]
        for ignore_label in self.ignore_labels:
            seg_map[seg_map == ignore_label] = 0
        seg_map_res = np.zeros_like(seg_map)
        seg_map_res[seg_map > 0] = 1
        seg_map_res[seg_map == self.ignore_ind] = 255
        results["gt_semantic_seg"] = seg_map_res
        return results


@PIPELINES.register_module()
class BinaryLabelDilate(object):
    """
    Dilate the binary label.
    Required keys are "gt_semantic_seg", added or modified keys are "gt_semantic_seg".
    Args:
        dilate_pixel (int): The pixel to dilate the binary label. Default: 3.
    """

    def __init__(self, dilate_pixel=3):
        self.dilate_pixel = dilate_pixel
        kernel_size = int(dilate_pixel * 2 + 1)
        self.conv_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )

    def __call__(self, results):
        """
        Call function to dilate the binary label.
        Args:
            results (dict): A result dict contains the gt_semantic_seg.
        Returns:
            results (dict): A result dict contains the gt_semantic_seg.
        """
        seg_map = results["gt_semantic_seg"]
        seg_map = cv2.dilate(seg_map, self.conv_kernel)
        results["gt_semantic_seg"] = seg_map
        return results


@PIPELINES.register_module()
class OmmitSmallRegion(object):
    """
    Ommit the small region in the input image.
    Required keys are "gt_semantic_seg", added or modified keys are "gt_semantic_seg".
    Args:
        min_area (int): The minimum area of the region to be kept. Default: 50.
    """

    def __init__(self, min_area=50):
        self.min_area = min_area

    def __call__(self, results):
        """
        Call function to ommit the small region in the input image.
        Args:
            results (dict): A result dict contains the gt_semantic_seg.
        Returns:
            results (dict): A result dict contains the gt_semantic_seg.
        """
        seg_map = results["gt_semantic_seg"]
        seg_map_binary = seg_map.copy()
        seg_map_binary[seg_map_binary > 0] = 1
        contours, _ = cv2.findContours(
            seg_map_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                cv2.drawContours(seg_map, [contour], -1, (0, 0, 0), -1)
        results["gt_semantic_seg"] = seg_map
        return results


@PIPELINES.register_module()
class DoubleImageRandomRgbBgr(object):
    """
    Randomly flip the RGB and BGR channels of the input images.
    Required keys are "img1" and "img2", added or modified keys are "img1" and "img2".
    Args:
        prob (float): Probability of flipping the channels. Default: 0.3.
    """

    def __init__(self, prob=0.3):
        """
        Constructor of the class.
        Args:
            prob (float): Probability of flipping the channels. Default: 0.3.
        """
        self.prob = prob

    def __call__(self, results):
        """
        Call function to randomly flip the RGB and BGR channels of the input images.
        Args:
            results (dict): A result dict contains the img1 and img2.
        Returns:
            results (dict): A result dict contains the img1 and img2.
        """
        if np.random.rand() < self.prob:
            results["img1"] = results["img1"][:, :, ::-1]
        if np.random.rand() < self.prob:
            results["img2"] = results["img2"][:, :, ::-1]
        return results


@PIPELINES.register_module()
class LoadAsMultiLabel(object):
    """
    Load the semantic segmentation label as multi-label.
    Required keys are "gt_semantic_seg", added or modified keys are "gt_semantic_seg_sat1" and "gt_semantic_seg_sat2".
    Args:
        ignore_bg (bool): Whether to ignore the background. Default: True.
    """

    def __init__(self, ignore_bg=True):
        CLASSES = [
            "绿地-动土",
            "动土-绿地",
            "硬化空地-动土",
            "动土-硬化空地",
            "绿地-硬化空地",
            "硬化空地-绿地",
            "动土-在建建筑",
            "绿地-在建建筑",
            "水域-在建建筑",
            "硬化空地-在建建筑",
            "绿地-建筑",
            "动土-建筑",
            "水域-建筑",
            "硬化空地-建筑",
            "建筑-建筑",
            "在建建筑-建筑",
            "在建建筑-绿地",
            "在建建筑-动土",
            "在建建筑-水域",
            "在建建筑-硬化空地",
            "建筑-绿地",
            "建筑-动土",
            "建筑-水域",
            "建筑-硬化空地",
            "建筑-在建建筑",
            "绿地-大棚",
            "动土-大棚",
            "水域-大棚",
            "硬化空地-大棚",
            "建筑-大棚",
            "在建建筑-大棚",
            "大棚-绿地",
            "大棚-水域",
            "大棚-动土",
            "大棚-在建建筑",
            "大棚-建筑",
            "大棚-硬化空地",
            "水域-动土",
            "水域-硬化空地",
            "动土-水域",
            "硬化空地-水域",
            "绿地-运动场",
            "水域-运动场",
            "动土-运动场",
            "硬化空地-运动场",
            "大棚-运动场",
            "建筑-运动场",
            "在建建筑-运动场",
            "运动场-绿地",
            "运动场-动土",
            "运动场-水域",
            "运动场-硬化空地",
            "运动场-建筑",
            "运动场-大棚",
            "运动场-在建建筑",
            "自然裸地-动土",
            "动土-自然裸地",
            "建筑-自然裸地",
            "自然裸地-建筑",
            "大棚-自然裸地",
            "自然裸地-大棚",
            "林地-在建建筑",
            "林地-建筑",
            "在建建筑-林地",
            "在建大棚-林地",
            "林地-动土",
            "林地-水域",
            "运动场-林地",
            "动土-林地",
            "地膜-林地",
            "建筑-林地",
            "林地-无作物耕地",
            "林地-在建大棚",
            "有作物耕地-林地",
            "林地-大棚",
            "林地-地膜",
            "硬化空地-林地",
            "无作物耕地-林地",
            "林地-有作物耕地",
            "水域-林地",
            "林地-运动场",
            "自然裸地-林地",
            "林地-硬化空地",
            "林地-自然裸地",
            "大棚-林地",
            "有作物耕地-动土",
            "无作物耕地-在建建筑",
            "无作物耕地-建筑",
            "在建建筑-无作物耕地",
            "有作物耕地-建筑",
            "在建建筑-有作物耕地",
            "硬化空地-无作物耕地",
            "有作物耕地-在建建筑",
            "无作物耕地-动土",
            "地膜-无作物耕地",
            "在建大棚-无作物耕地",
            "建筑-无作物耕地",
            "运动场-无作物耕地",
            "动土-有作物耕地",
            "无作物耕地-在建大棚",
            "有作物耕地-地膜",
            "大棚-无作物耕地",
            "大棚-有作物耕地",
            "硬化空地-有作物耕地",
            "动土-无作物耕地",
            "无作物耕地-大棚",
            "有作物耕地-大棚",
            "有作物耕地-在建大棚",
            "无作物耕地-地膜",
            "有作物耕地-硬化空地",
            "有作物耕地-运动场",
            "在建大棚-有作物耕地",
            "无作物耕地-运动场",
            "地膜-有作物耕地",
            "建筑-有作物耕地",
            "无作物耕地-硬化空地",
            "运动场-有作物耕地",
        ]
        categories = set()
        for cats in CLASSES:
            categories.update(set(cats.split("-")))
        categories = list(categories)
        categories = sorted(categories)
        label_mapping = [[255, 255]]
        for cats in CLASSES:
            cat1, cat2 = cats.split("-")
            label_mapping.append([categories.index(cat1), categories.index(cat2)])
        self.label_mapping = np.array(label_mapping).astype(np.uint8)
        self.ignore_bg = ignore_bg

    def __call__(self, results):
        seg_map = results["gt_semantic_seg"]
        multi_cls_seg_map = self.label_mapping[seg_map]
        gt_semantic_seg_sat1 = multi_cls_seg_map[:, :, 0]
        gt_semantic_seg_sat2 = multi_cls_seg_map[:, :, 1]
        if not self.ignore_bg:
            gt_semantic_seg_sat1 += 1
            gt_semantic_seg_sat2 += 1
        if "seg_fields" in results:
            results["seg_fields"].extend(
                ["gt_semantic_seg_sat1", "gt_semantic_seg_sat2"]
            )
        results["gt_semantic_seg_sat1"] = gt_semantic_seg_sat1
        results["gt_semantic_seg_sat2"] = gt_semantic_seg_sat2
        return results


@PIPELINES.register_module()
class LoadAs19ClassLabel(object):
    """
    LoadAs19ClassLabel pipeline that maps the original 85-class semantic segmentation labels to 19 classes.
    Args:
        ignore_bg (bool): Whether to ignore the background class (class 0) when mapping labels. Default: False.
        raw_classes (list[str]): List of original 85 classes. If not provided, the default list of classes will be used.
        index2change (dict): Dictionary that maps the index of a class in the original 85-class list to the index of the
            corresponding class in the 19-class list. If not provided, the default mapping will be used.
    """

    def __init__(self, ignore_bg=False, raw_classes=None, index2change=None):
        CLASSES = raw_classes
        INDEX2CHANGE = index2change
        if CLASSES is None:
            CLASSES = [
                "绿地-动土",
                "动土-绿地",
                "硬化空地-动土",
                "动土-硬化空地",
                "绿地-硬化空地",
                "硬化空地-绿地",
                "动土-在建建筑",
                "绿地-在建建筑",
                "水域-在建建筑",
                "硬化空地-在建建筑",
                "绿地-建筑",
                "动土-建筑",
                "水域-建筑",
                "硬化空地-建筑",
                "建筑-建筑",
                "在建建筑-建筑",
                "在建建筑-绿地",
                "在建建筑-动土",
                "在建建筑-水域",
                "在建建筑-硬化空地",
                "建筑-绿地",
                "建筑-动土",
                "建筑-水域",
                "建筑-硬化空地",
                "建筑-在建建筑",
                "绿地-大棚",
                "动土-大棚",
                "水域-大棚",
                "硬化空地-大棚",
                "建筑-大棚",
                "在建建筑-大棚",
                "大棚-绿地",
                "大棚-水域",
                "大棚-动土",
                "大棚-在建建筑",
                "大棚-建筑",
                "大棚-硬化空地",
                "水域-动土",
                "水域-硬化空地",
                "动土-水域",
                "硬化空地-水域",
                "绿地-运动场",
                "水域-运动场",
                "动土-运动场",
                "硬化空地-运动场",
                "大棚-运动场",
                "建筑-运动场",
                "在建建筑-运动场",
                "运动场-绿地",
                "运动场-动土",
                "运动场-水域",
                "运动场-硬化空地",
                "运动场-建筑",
                "运动场-大棚",
                "运动场-在建建筑",
                "自然裸地-动土",
                "动土-自然裸地",
                "建筑-自然裸地",
                "自然裸地-建筑",
                "大棚-自然裸地",
                "自然裸地-大棚",
                "林地-在建建筑",
                "林地-建筑",
                "在建建筑-林地",
                "在建大棚-林地",
                "林地-动土",
                "林地-水域",
                "运动场-林地",
                "动土-林地",
                "地膜-林地",
                "建筑-林地",
                "林地-无作物耕地",
                "林地-在建大棚",
                "有作物耕地-林地",
                "林地-大棚",
                "林地-地膜",
                "硬化空地-林地",
                "无作物耕地-林地",
                "林地-有作物耕地",
                "水域-林地",
                "林地-运动场",
                "自然裸地-林地",
                "林地-硬化空地",
                "林地-自然裸地",
                "大棚-林地",
                "有作物耕地-动土",
                "无作物耕地-在建建筑",
                "无作物耕地-建筑",
                "在建建筑-无作物耕地",
                "有作物耕地-建筑",
                "在建建筑-有作物耕地",
                "硬化空地-无作物耕地",
                "有作物耕地-在建建筑",
                "无作物耕地-动土",
                "地膜-无作物耕地",
                "在建大棚-无作物耕地",
                "建筑-无作物耕地",
                "运动场-无作物耕地",
                "动土-有作物耕地",
                "无作物耕地-在建大棚",
                "有作物耕地-地膜",
                "大棚-无作物耕地",
                "大棚-有作物耕地",
                "硬化空地-有作物耕地",
                "动土-无作物耕地",
                "无作物耕地-大棚",
                "有作物耕地-大棚",
                "有作物耕地-在建大棚",
                "无作物耕地-地膜",
                "有作物耕地-硬化空地",
                "有作物耕地-运动场",
                "在建大棚-有作物耕地",
                "无作物耕地-运动场",
                "地膜-有作物耕地",
                "建筑-有作物耕地",
                "无作物耕地-硬化空地",
                "运动场-有作物耕地",
            ]
        if INDEX2CHANGE is None:
            INDEX2CHANGE = {
                0: [
                    "自然裸地-地膜",
                    "自然裸地-有作物耕地",
                    "无作物耕地-有作物耕地",
                    "地膜-建筑",
                    "绿地-地膜",
                    "自然裸地-无作物耕地",
                    "地膜-硬化空地",
                    "水域-地膜",
                    "绿地-有作物耕地",
                    "有作物耕地-地膜",
                    "在建大棚-大棚",
                    "有作物耕地-绿地",
                    "林地-地膜",
                    "自然裸地-绿地",
                    "无作物耕地-地膜",
                    "有作物耕地-无作物耕地",
                    "绿地-无作物耕地",
                    "无作物耕地-自然裸地",
                    "在建建筑-地膜",
                    "地膜-动土",
                    "大棚-地膜",
                    "林地-无作物耕地",
                    "地膜-自然裸地",
                    "无作物耕地-林地",
                    "无作物耕地-绿地",
                    "地膜-绿地",
                    "地膜-无作物耕地",
                    "地膜-在建建筑",
                    "有作物耕地-自然裸地",
                    "自然裸地-林地",
                    "硬化空地-地膜",
                    "绿地-自然裸地",
                    "大棚-在建大棚",
                    "地膜-有作物耕地",
                    "林地-有作物耕地",
                    "林地-自然裸地",
                    "绿地-林地",
                    "地膜-林地",
                    "有作物耕地-林地",
                    "建筑-地膜",
                    "动土-地膜",
                    "运动场-地膜",
                    "在建大棚-地膜",
                    "地膜-运动场",
                    "地膜-水域",
                    "地膜-在建大棚",
                    "地膜-大棚",
                ],
                1: [
                    "动土-在建建筑",
                    "动土-建筑",
                    "运动场-在建建筑",
                    "运动场-建筑",
                    "自然裸地-建筑",
                    "硬化空地-在建建筑",
                    "硬化空地-建筑",
                    "自然裸地-在建建筑",
                    "无作物耕地-建筑",
                    "无作物耕地-在建建筑",
                ],
                2: [
                    "在建建筑-动土",
                    "建筑-动土",
                    "在建建筑-运动场",
                    "建筑-运动场",
                    "建筑-自然裸地",
                    "在建建筑-硬化空地",
                    "建筑-硬化空地",
                    "建筑-无作物耕地",
                    "在建建筑-自然裸地",
                ],
                3: ["绿地-在建建筑", "绿地-建筑", "林地-在建建筑", "有作物耕地-在建建筑", "林地-建筑", "有作物耕地-建筑"],
                4: [
                    "在建建筑-绿地",
                    "建筑-绿地",
                    "在建建筑-林地",
                    "建筑-林地",
                    "建筑-有作物耕地",
                    "在建建筑-有作物耕地",
                    "在建建筑-无作物耕地",
                ],
                5: ["动土-大棚", "硬化空地-大棚", "自然裸地-大棚", "在建建筑-大棚", "建筑-大棚", "水域-大棚"],
                6: ["大棚-动土", "大棚-硬化空地", "大棚-自然裸地", "大棚-在建建筑", "大棚-建筑", "大棚-水域"],
                7: [
                    "绿地-大棚",
                    "运动场-大棚",
                    "无作物耕地-大棚",
                    "在建建筑-在建大棚",
                    "水域-在建大棚",
                    "林地-在建大棚",
                    "建筑-在建大棚",
                    "运动场-在建大棚",
                    "动土-在建大棚",
                    "林地-大棚",
                    "绿地-在建大棚",
                    "自然裸地-在建大棚",
                    "无作物耕地-在建大棚",
                    "有作物耕地-在建大棚",
                    "硬化空地-在建大棚",
                    "有作物耕地-大棚",
                ],
                8: [
                    "大棚-绿地",
                    "大棚-运动场",
                    "在建大棚-绿地",
                    "在建大棚-自然裸地",
                    "大棚-有作物耕地",
                    "在建大棚-有作物耕地",
                    "在建大棚-动土",
                    "大棚-林地",
                    "在建大棚-林地",
                    "在建大棚-硬化空地",
                    "大棚-无作物耕地",
                    "在建大棚-无作物耕地",
                    "在建大棚-建筑",
                    "在建大棚-运动场",
                    "在建大棚-在建建筑",
                    "在建大棚-水域",
                ],
                9: ["动土-水域", "自然裸地-水域", "硬化空地-水域", "运动场-水域", "无作物耕地-水域"],
                10: ["水域-动土", "水域-自然裸地", "水域-硬化空地", "水域-运动场", "水域-无作物耕地"],
                11: ["动土-硬化空地", "动土-自然裸地", "动土-无作物耕地"],
                12: ["硬化空地-动土", "自然裸地-动土", "无作物耕地-动土"],
                13: [
                    "动土-绿地",
                    "硬化空地-绿地",
                    "动土-运动场",
                    "硬化空地-运动场",
                    "运动场-绿地",
                    "硬化空地-无作物耕地",
                    "硬化空地-林地",
                    "动土-林地",
                    "自然裸地-运动场",
                    "硬化空地-有作物耕地",
                    "运动场-有作物耕地",
                    "动土-有作物耕地",
                    "运动场-林地",
                    "硬化空地-自然裸地",
                ],
                14: [
                    "绿地-动土",
                    "绿地-硬化空地",
                    "运动场-动土",
                    "运动场-硬化空地",
                    "绿地-运动场",
                    "有作物耕地-动土",
                    "自然裸地-硬化空地",
                    "运动场-无作物耕地",
                    "无作物耕地-运动场",
                    "林地-硬化空地",
                    "有作物耕地-硬化空地",
                    "运动场-自然裸地",
                    "林地-运动场",
                    "无作物耕地-硬化空地",
                    "林地-动土",
                    "有作物耕地-运动场",
                ],
                15: ["在建建筑-水域", "建筑-水域"],
                16: ["水域-在建建筑", "水域-建筑"],
                17: ["水域-绿地", "水域-有作物耕地", "水域-林地"],
                18: ["绿地-水域", "有作物耕地-水域", "林地-水域"],
                19: ["在建建筑-建筑", "建筑-在建建筑", "建筑-建筑", "在建建筑-在建建筑"],
            }
        if ignore_bg:
            label_mapping = [255]
            for cat in CLASSES:
                for key, value in INDEX2CHANGE.items():
                    if cat in value:
                        label_mapping.append(np.uint8(key - 1))
                        break
        else:
            label_mapping = [0]
            for cat in CLASSES:
                for key, value in INDEX2CHANGE.items():
                    if cat in value:
                        label_mapping.append(key)
                        break

        assert len(label_mapping) == len(CLASSES) + 1
        self.label_mapping = np.array(label_mapping).astype(np.uint8)

    def __call__(self, results):
        seg_map = results["gt_semantic_seg"]
        seg_map_multi = self.label_mapping[seg_map]
        if "seg_fields" in results:
            results["seg_fields"].extend(["gt_semantic_seg_multi"])
        results["gt_semantic_seg_multi"] = seg_map_multi
        return results


@PIPELINES.register_module()
class LoadAsBinaryEdges(object):
    """
    LoadAsBinaryEdges: a pipeline to load binary edges from semantic segmentation.
    """

    def __init__(self, radius=2, dilate_pixel=2, outside=False):
        """
        Args:
            radius (int): radius of the edges.
            dilate_pixel (int): pixel to dilate.
            outside (bool): whether to load edges outside the object.
        """
        self.radius = radius
        self.dilate_pixel = dilate_pixel
        self.outside = outside
        kernel_size = int(dilate_pixel * 2 + 1)
        self.conv_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )

    def __call__(self, results):
        """
        Load binary edges from semantic segmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: The result dict with binary edges.
        """
        seg_map = results["gt_semantic_seg"]
        edges = np.zeros_like(seg_map, dtype=np.uint8)
        num_class = seg_map.max()
        for i_cls in range(1, num_class + 1):
            mask = (seg_map == i_cls).astype(np.uint8)
            if mask.sum() == 0:
                continue
            dist = distance_transform_edt(1 - mask)
            if self.outside:
                dist[dist == 0] = 99999
            else:
                dist += distance_transform_edt(mask)
            # dist[dist <= self.radius] = 1
            # dist[dist > self.radius] = 0
            # edges += dist.astype(np.uint8)
            edges[dist <= self.radius] += 1
        edges[edges > 0] = 1
        # _edges = edges.copy()
        # _edges[seg_map == 0] = 0
        # _edges = cv2.dilate(_edges, self.conv_kernel)
        # edges[seg_map == 0] = 255
        # edges[_edges == 1] = 1

        results["gt_edges"] = edges
        return results


@PIPELINES.register_module()
class RandomReverse(object):
    """
    Randomly reverse the order of two images and their corresponding semantic segmentation maps.

    Args:
        prob (float): Probability of reversing the order.

    Returns:
        dict: The result dict with reversed images and semantic segmentation maps.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() < self.prob:
            if "gt_semantic_seg_sat1" in results:
                results["gt_semantic_seg_sat1"], results["gt_semantic_seg_sat2"] = (
                    results["gt_semantic_seg_sat2"],
                    results["gt_semantic_seg_sat1"],
                )
            results["img1"], results["img2"] = results["img2"], results["img1"]
        return results


@PIPELINES.register_module()
class RandomResizeDownUp(object):
    """
    Randomly resize the input images with a probability of 0.6.
    If the image is resized, it will be either resized to 0.5x or 4x of its original size.
    Args:
        None
    Returns:
        dict: The result dict with resized images.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        results["img1"] = self._random_resize(results["img1"])
        results["img2"] = self._random_resize(results["img2"])
        return results

    def _random_resize(self, img):
        h, w, c = img.shape
        if np.random.rand() < 0.6:
            r1 = np.random.rand()
            if r1 < 0.5:
                s1 = mmcv.imrescale(img, 0.5)
                img = mmcv.imrescale(s1, 2.0)
            else:
                s1 = mmcv.imrescale(img, 0.25)
                img = mmcv.imrescale(s1, 4.0)

        return img


@PIPELINES.register_module()
class RandomShift(object):
    """
    Shift the input images randomly with a probability of `prob`.
    If the image is shifted, it will be shifted by a random number of pixels between `-max_pixel` and `max_pixel`.
    Args:
        prob (float): Probability of shifting the image.
        max_pixel (int): Maximum number of pixels to shift the image.
        return_shift (bool): Whether to return the shift amount as a key in the results dictionary.
        method (str): Method to use for generating the shift amount. Can be either 'uniform' or 'normal'.
    Returns:
        dict: The result dict with shifted images and semantic segmentation maps.
    """

    def __init__(self, prob=0.5, max_pixel=5, return_shift=False, method="uniform"):
        self.prob = prob
        self.max_pixel = max_pixel
        self.return_shift = return_shift
        assert method in ["uniform", "normal"]
        self.method = method

    def __call__(self, results):
        if np.random.rand() < self.prob:
            if self.method == "uniform":
                shiftx = np.random.randint(-self.max_pixel, self.max_pixel + 1)
                shifty = np.random.randint(-self.max_pixel, self.max_pixel + 1)
            elif self.method == "normal":
                shiftx = int(
                    np.clip(np.random.normal(scale=0.4), -1, 1) * self.max_pixel
                )
                shifty = int(
                    np.clip(np.random.normal(scale=0.4), -1, 1) * self.max_pixel
                )
            img1 = results["img1"]
            rows, cols = img1.shape[:2]

            M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
            img1 = cv2.warpAffine(img1, M, (cols, rows))
            results["img1"] = img1
        else:
            shiftx, shifty = 0, 0
        if self.return_shift:
            results["shift"] = -np.array([shiftx, shifty], dtype=float)
        seg_map = results.get("gt_semantic_seg", None)
        if not seg_map is None:
            if shiftx > 0:
                seg_map[:, :shiftx] = 0
            if shiftx < 0:
                seg_map[:, shiftx:] = 0
            if shifty > 0:
                seg_map[:shifty, :] = 0
            if shifty < 0:
                seg_map[shifty:, :] = 0
            results["gt_semantic_seg"] = seg_map
        return results


@PIPELINES.register_module()
class Shift(object):
    """
    Shift the input images by a fixed number of pixels.
    Args:
        pixel (int): Number of pixels to shift the image.
        return_shift (bool): Whether to return the shift amount as a key in the results dictionary.
    Returns:
        dict: The result dict with shifted images and semantic segmentation maps.
    """

    def __init__(self, pixel=5, return_shift=False):
        self.pixel = pixel
        self.return_shift = return_shift

    def __call__(self, results):
        shiftx = self.pixel
        shifty = self.pixel
        img1 = results["img1"]
        rows, cols = img1.shape[:2]

        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        img1 = cv2.warpAffine(img1, M, (cols, rows))
        results["img1"] = img1
        if self.return_shift:
            results["shift"] = -np.array([shiftx, shifty], dtype=float)


@PIPELINES.register_module()
class ConvertMCDLabel(object):
    def __init__(
        self,
        danqi_label=False,
        annotation_type="v25",
        mcd_class_name=None,
        mcd_class_id=None,
    ):
        if annotation_type == "custom":
            self.get_custom_mapping(mcd_class_name, mcd_class_id)
        elif annotation_type == "yuzhi":
            self.get_yuzhi_mapping()
        else:
            print("Invalid annotation_type")
            return
        self.danqi_label = danqi_label
        # NOTE: share convert MCD here, mapping is very important!!!

    def get_sense_mapping(self):
        """Get Sense mapping."""

        fg_class_num = 6
        label_mapping, pre_mapping, post_mapping = [], [], []
        self.label_mapping = np.arange(1 + fg_class_num * fg_class_num)
        self.pre_mapping = (self.label_mapping - 1) // fg_class_num
        self.post_mapping = (self.label_mapping - 1) % fg_class_num
        self.pre_mapping[0] = 255
        self.post_mapping[0] = 255

    def get_custom_mapping(self, class_name, class_id):
        """Get Sense mapping."""

        basic_class = []
        for item in class_name[1:]:
            pre_name, post_name = item.split("-")
            basic_class += [pre_name, post_name]
        self.BASIC_CLASS = sorted(list(np.unique(basic_class)))
        basic_class = self.BASIC_CLASS
        pre_mapping, post_mapping = [255], [255]
        for item in class_name[1:]:
            pre_name, post_name = item.split("-")
            pre_index = self.BASIC_CLASS.index(pre_name)
            post_index = self.BASIC_CLASS.index(post_name)
            pre_mapping.append(pre_index)
            post_mapping.append(post_index)
        self.pre_mapping = np.array(pre_mapping)
        self.post_mapping = np.array(post_mapping)
        pad_num = 256 - len(self.pre_mapping)
        self.pre_mapping = np.pad(self.pre_mapping, (0, pad_num), constant_values=255)
        self.post_mapping = np.pad(self.post_mapping, (0, pad_num), constant_values=255)
        # import pickle
        # mcd_meta = {'fg_class_num': len(self.BASIC_CLASS)}
        # pickle.dump(open('mcd_meta.pkl', 'wb'), mcd_meta)

        remapping_label = []
        for pre_index, pre_name in enumerate(basic_class):
            for post_index, post_name in enumerate(basic_class):
                combined_name = f"{pre_name}-{post_name}"
                if combined_name in class_name:
                    remapping_label.append(class_name.index(combined_name))
                else:
                    remapping_label.append(0)
        # print(basic_class, class_name)
        # print(remapping_label)
        self.label_mapping = np.array(remapping_label)

    def get_yuzhi_mapping(self):
        """Get Sense mapping."""

        self.BASIC_CLASS = ["耕地", "绿地", "建筑", "动土", "自然裸地", "水体", "大棚", "构筑物"]
        fg_class_num = 8
        self.label_mapping = np.arange(1 + fg_class_num * fg_class_num)
        self.pre_mapping = (self.label_mapping - 1) // fg_class_num
        self.post_mapping = (self.label_mapping - 1) % fg_class_num
        self.pre_mapping[0] = 255
        self.post_mapping[0] = 255
        pad_num = 256 - len(self.label_mapping)
        self.pre_mapping = np.pad(self.pre_mapping, (0, pad_num), constant_values=255)
        self.post_mapping = np.pad(self.post_mapping, (0, pad_num), constant_values=255)

    def __call__(self, results):
        # results['gt_semantic_seg'] = self.label_mapping[seg_map]
        seg_map = results["gt_semantic_seg"]
        if self.danqi_label:
            pre_map = self.pre_mapping[seg_map]
            post_map = self.post_mapping[seg_map]
            results["gt_semantic_seg_sat1"] = pre_map
            results["gt_semantic_seg_sat2"] = post_map
            # NOTE: very important this debug case
            # import numpy as np
            # for item in ['gt_semantic_seg_sat1', 'gt_semantic_seg_sat2', 'gt_semantic_seg']:
            #     temp = results[item]
            #     print(np.unique(temp, return_counts=True))
            #     indexes = temp > 0
            #     temp[indexes] += 125
            #     cv2.imwrite(f'./debug/{item}.png', temp.astype('uint8'))
        return results


@PIPELINES.register_module()
class SplitMCDLabel(object):
    def __init__(self, fg_class_num):
        self.fg_class_num = fg_class_num

    def separate_map(self, combined_map):
        """Separate map."""

        combined_map = combined_map.copy()  # [0, 64]
        mask = combined_map == 0
        combined_map -= 1  # [-1, 63]
        prev = combined_map // self.fg_class_num  # [0, 7]
        post = combined_map % self.fg_class_num  # [0, 7]
        prev += 1  # [1, 8]
        post += 1
        prev[mask] = 0
        post[mask] = 0
        return prev, post

    def __call__(self, results):
        seg_map = results["gt_semantic_seg"].copy()
        # convert label to binary
        gt_semantic_seg = results["gt_semantic_seg"]
        gt_semantic_seg[gt_semantic_seg > 0] = 1
        results["gt_semantic_seg"] = gt_semantic_seg
        prev, post = self.separate_map(seg_map)
        results["gt_semantic_seg_sat1"] = prev
        results["gt_semantic_seg_sat2"] = post

        return results


@PIPELINES.register_module()
class GSDNorm(object):
    """
    Normalize the GSD of the image to a certain value.

    Args:
        gsd_info_file (str): The path of the file that contains the GSD information.
        norm_gsd (float): The normalized GSD value.
        max_ratio (int): The maximum ratio of the GSD value to the normalized GSD value.
    """

    def __init__(self, gsd_info_file, norm_gsd=1.0, max_ratio=3):
        self.norm_gsd = norm_gsd
        self.gsd_map = self.get_gsd_info(gsd_info_file)
        self.max_ratio = max_ratio

    def get_gsd_info(self, gsd_info_file):
        """
        Get the GSD information from the file.

        Args:
            gsd_info_file (str): The path of the file that contains the GSD information.

        Returns:
            dict: A dictionary that maps the base name of the image to its GSD value.
        """
        gsd_list = [0.15, 0.5, 0.8, 1.0, 1.5, 2.0]
        with open(gsd_info_file, "r") as file:
            data = file.readlines()
        gsd_map = {}
        for ele in data:
            gsd, base_name = ele.replace("\n", "").split("\t")
            gsd = float(gsd)
            base_name = base_name.replace("_sat.npy", "").split("_")[0]
            assert gsd in gsd_list
            gsd_map[base_name] = gsd
        return gsd_map

    def __call__(self, results):
        """
        Call function to normalize the GSD of the image.

        Args:
            results (dict): The input data dictionary.

        Returns:
            dict: The output data dictionary.
        """
        base_name = results["filename"][0].split("/")[-1].split("_")[0]
        gsd = self.gsd_map.get(base_name, self.norm_gsd)
        ratio = gsd / self.norm_gsd
        ratio = np.clip(ratio, 1 / self.max_ratio, self.max_ratio)
        h, w = results["img1"].shape[:2]
        scale = int(w * ratio), int(h * ratio)
        img1, scale_factor = mmcv.imrescale(results["img1"], scale, return_scale=True)
        img2, scale_factor = mmcv.imrescale(results["img2"], scale, return_scale=True)

        for key in results.get("seg_fields", []):
            gt_seg = mmcv.imrescale(results[key], scale, interpolation="nearest")
            results[key] = gt_seg
        return results


@PIPELINES.register_module()
class DoubleImageRandomBlur(object):
    """
    Pipeline for randomly blurring images.

    Args:
        prob (float): The probability of applying the blur.
        kernel_size (int): The size of the kernel for the Gaussian blur.
    """

    def __init__(self, prob=0.3, kernel_size=3):
        self.prob = prob
        self.kernel_size = kernel_size

    def __call__(self, results):
        """
        Call function to apply the blur.

        Args:
            results (dict): The input data dictionary.

        Returns:
            dict: The output data dictionary.
        """
        for key in ["img1", "img2"]:
            if np.random.rand() < self.prob:
                img = results[key]
                img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
                results[key] = img
        return results


@PIPELINES.register_module()
class DoubleImageRandomFog(object):
    """
    Pipeline for randomly adding fog to images.

    Args:
        prob (float): The probability of applying the fog.
        size (int): The size of the fog.
        random_color (bool): Whether to randomly change the color of the fog.
        thin (bool): Whether to make the fog thinner.
    """

    def __init__(self, prob=0.05, size=400, random_color=False, thin=False):
        import imgaug.augmenters as iaa

        self.aug = iaa.Clouds()
        self.prob = prob
        self.size = size
        self.random_color = random_color
        self.thin = thin

    def __call__(self, results):
        if np.random.rand() < self.prob:
            if np.random.rand() < 0.5:
                aug_name = "img1"
            else:
                aug_name = "img2"
            img = results[aug_name]
            h, w, c = img.shape
            size = min(min(h, w), self.size)
            y = np.random.randint(h - size + 1)
            x = np.random.randint(w - size + 1)
            img_aug = self.aug.augment_image(img)
            img_aug = self.aug.augment_image(img_aug)
            if self.thin:
                if np.random.rand() < 0.5:
                    img_aug = self.aug.augment_image(img_aug)
            else:
                img_aug = self.aug.augment_image(img_aug)
            if self.random_color:
                img_aug = img_aug.astype(int)
                img_aug -= ((np.random.rand(3)) * 70).astype(int)
                img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)
            img[y : y + size, x : x + size] = img_aug[y : y + size, x : x + size]
            results[aug_name] = img
            for key in results.get("seg_fields", []):
                gt_seg = results[key]
                gt_seg[y : y + size, x : x + size] = 0
                results[key] = gt_seg

        return results


@PIPELINES.register_module()
class DoubleImageWatermark(object):
    """
    Pipeline for randomly adding watermarks to images.

    Args:
        png_list (list): A list of paths to the watermark images.
        prob (float): The probability of applying the watermark.
        min_size (int): The minimum size of the watermark.
        max_size (int): The maximum size of the watermark.
    """

    def __init__(self, png_list, prob=0.5, min_size=60, max_size=100):
        self.watermarks = []
        for png in png_list:
            self.watermarks.append(cv2.imread(png))
        self.prob = prob
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, results):
        """
        Call function to apply the watermark.

        Args:
            results (dict): The input data dictionary.

        Returns:
            dict: The output data dictionary.
        """
        if np.random.rand() < self.prob:
            if np.random.rand() < 0.5:
                aug_name = "img1"
            else:
                aug_name = "img2"
            img = results[aug_name]
            watermark = self.watermarks[np.random.randint(len(self.watermarks))]
            size = np.random.randint(self.min_size, self.max_size)
            watermark = cv2.resize(watermark, (size, size))
            h, w, c = img.shape
            y = np.random.randint((h - size))
            x = np.random.randint(w - size)
            img_tmp = img[y : y + size, x : x + size].astype(int)
            if np.random.rand() > 0.5:
                img_tmp += watermark // np.random.randint(2, 6)
            else:
                _watermark = (watermark > 0).astype(int) * 190
                noise = ((np.random.rand(*(_watermark.shape)) - 0.5) * 30).astype(int)
                _watermark[_watermark > 0] = (_watermark + noise)[_watermark > 0]
                img_tmp[_watermark > 0] = _watermark[_watermark > 0]
            img_tmp = np.clip(img_tmp, 0, 255).astype(np.uint8)
            img[y : y + size, x : x + size] = img_tmp
            results[aug_name] = img
        return results


@PIPELINES.register_module()
class SmallTargetAug(object):
    """
    Pipeline for augmenting small targets.

    Args:
        txtfile (str): The path of the file that contains the information of the small targets.
        patch_size (int): The size of the patch.
        prob (float): The probability of applying the augmentation.
        min_area (int): The minimum area of the small targets.
        max_area (int): The maximum area of the small targets.
        aug_cats (list): The categories of the small targets to be augmented.
        sample_method (str): The method of sampling the small targets.
    """

    def __init__(
        self,
        txtfile,
        patch_size=200,
        prob=0.2,
        min_area=0,
        max_area=400,
        aug_cats=None,
        sample_method="random",
    ):
        with open(txtfile, "r") as file:
            files = file.readlines()
        self.samples = []
        for sample in files:
            _, _, _, cat, area = sample.replace("\n", "").split(" ")
            cat = int(cat)
            area = float(area)
            if area < max_area and area > min_area:
                if aug_cats is None or cat in aug_cats:
                    self.samples.append(sample)
        self.patch_size = patch_size
        self.prob = prob
        self.sample_method = sample_method
        if sample_method == "queue":
            np.random.shuffle(self.samples)
            self.index = 0

    def __call__(self, results):
        """
        Call function to augment the small targets.

        Args:
            results (dict): The input data dictionary.

        Returns:
            dict: The output data dictionary.
        """
        if np.random.rand() < self.prob:
            img = results["img1"]
            h, w, c = img.shape
            y = np.random.randint(h - self.patch_size)
            x = np.random.randint(w - self.patch_size)
            if self.sample_method == "queue":
                sample = self.samples[self.index % len(self.samples)]
                self.index += 1
            else:
                sample = np.random.choice(self.samples)
            patch1, patch2, patch_mask, _, _ = sample.replace("\n", "").split(" ")
            patch1 = cv2.imread(patch1)
            patch2 = cv2.imread(patch2)
            patch_mask = cv2.imread(patch_mask, -1)

            img1 = results["img1"]
            img2 = results["img2"]
            img1[y : y + self.patch_size, x : x + self.patch_size, :] = patch1
            img2[y : y + self.patch_size, x : x + self.patch_size, :] = patch2
            results["img1"] = img1
            results["img2"] = img2

            for key in results.get("seg_fields", []):
                gt_seg = results[key]
                gt_seg[y : y + self.patch_size, x : x + self.patch_size] = patch_mask
                results[key] = gt_seg

        return results


@PIPELINES.register_module()
class GridShuffle(object):
    """
    Pipeline for shuffling the grid of the image.

    Args:
        grid_num (int): The number of grids.
        img_size (int): The size of the image.
    """

    def __init__(self, grid_num=3, img_size=896):
        self.grid_size = int(np.ceil(img_size / grid_num))
        self.pad_size = int(self.grid_size * grid_num)
        self.padding = int(self.pad_size - img_size)
        self.img_size = img_size
        self.grid_num = grid_num
        self.num = self.grid_num**2

    def __call__(self, results):
        """
        Call function to shuffle the grid of the image.

        Args:
            results (dict): The input data dictionary.

        Returns:
            dict: The output data dictionary.
        """
        grid_ind = np.arange(self.num)
        np.random.shuffle(grid_ind)

        for key in results.get("seg_fields", []) + ["img1", "img2"]:
            data = results[key]
            if len(data.shape) == 3:
                data = np.pad(data, [[0, self.padding], [0, self.padding], [0, 0]])
            else:
                data = np.pad(data, [[0, self.padding], [0, self.padding]])
            data_new = np.zeros_like(data)
            ind = 0
            for i in range(self.grid_num):
                for j in range(self.grid_num):
                    row = grid_ind[ind] // self.grid_num
                    col = grid_ind[ind] % self.grid_num
                    data_new[
                        row * self.grid_size : (row + 1) * self.grid_size,
                        col * self.grid_size : (col + 1) * self.grid_size,
                    ] = data[
                        i * self.grid_size : (i + 1) * self.grid_size,
                        j * self.grid_size : (j + 1) * self.grid_size,
                    ]
                    ind += 1
            results[key] = data_new[: self.img_size, : self.img_size]
        return results
