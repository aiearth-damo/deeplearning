# -*- conding: utf-8 -*-
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines import Compose, LoadAnnotations


@DATASETS.register_module()
class LandcoverLoader(CustomDataset):
    """LandcoverLoader dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.

    Args:
        pipeline (list[dict]): A sequence of data transforms.
        img_dir (str): The directory of the images.
        img_suffix (str): Suffix of the images.
        ann_dir (str): The directory of the annotations.
        seg_map_suffix (str): Suffix of the segmentation maps.
        split (str): The split file of the dataset.
        data_root (str): The root directory of the dataset.
        test_mode (bool): If True, the dataset is for testing.
        ignore_index (int): The index that will be ignored in the segmentation maps.
        reduce_zero_label (bool): If True, the label 0 will be ignored.
        classes (int): The number of classes in the dataset.
        palette (list[list[int]]): The color palette of the dataset.
        gt_seg_map_loader_cfg (dict): Configurations for loading the segmentation maps.

    Attributes:
        classes (list[str]): The list of class names.
        pipeline (Compose): The data pipeline.
        img_dir (str): The directory of the images.
        img_suffix (str): Suffix of the images.
        ann_dir (str): The directory of the annotations.
        seg_map_suffix (str): Suffix of the segmentation maps.
        split (str): The split file of the dataset.
        data_root (str): The root directory of the dataset.
        test_mode (bool): If True, the dataset is for testing.
        ignore_index (int): The index that will be ignored in the segmentation maps.
        reduce_zero_label (bool): If True, the label 0 will be ignored.
        label_map (None): The label map of the dataset.
        CLASSES (list[str]): The list of class names.
        PALETTE (list[list[int]]): The color palette of the dataset.
        gt_seg_map_loader (LoadAnnotations): The loader for the segmentation maps.
        img_infos (list[dict]): The information of the images.

    """

    def __init__(
        self,
        pipeline,
        img_dir,
        img_suffix=".jpg",
        ann_dir=None,
        seg_map_suffix=".png",
        split=None,
        data_root=None,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        gt_seg_map_loader_cfg=None,
    ):
        # tmp_classes = []
        # for i in range(int(classes)):
        #    tmp_classes.append(str(i))
        # classes = tmp_classes
        self.classes = classes
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)
        self.gt_seg_map_loader = (
            LoadAnnotations()
            if gt_seg_map_loader_cfg is None
            else LoadAnnotations(**gt_seg_map_loader_cfg)
        )

        if test_mode:
            assert (
                self.CLASSES is not None
            ), "`cls.CLASSES` or `classes` should be specified when testing"

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(
            self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split
        )

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir
        results["seg_prefix"] = self.ann_dir
        results["img_infos"] = self.img_infos
        if self.custom_classes:
            results["label_map"] = self.label_map

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load the annotations of the dataset.

        Args:
            img_dir (str): The directory of the images.
            img_suffix (str): Suffix of the images.
            ann_dir (str): The directory of the annotations.
            seg_map_suffix (str): Suffix of the segmentation maps.
            split (str): The split file of the dataset.

        Returns:
            list[dict]: The information of the images.

        """
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    label_pair = line.strip().split(" ")
                    if len(label_pair) == 1:
                        img_name = label_pair[0]
                        img_info = dict(filename=img_name + img_suffix)
                        if ann_dir is not None:
                            seg_map = img_name + seg_map_suffix
                            img_info["ann"] = dict(seg_map=seg_map)
                        img_infos.append(img_info)
                    else:
                        img_info = dict(filename=label_pair[0])
                        if len(label_pair) > 1:
                            img_info["ann"] = dict(seg_map=label_pair[1])
                        img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f"Loaded {len(img_infos)} images", logger="root")
        return img_infos

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.

        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            filename = self.img_infos[idx]["filename"]
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f"{basename}.png")

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self, results, imgfile_prefix, to_label_id=True, indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.

        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), "results must be a list."
        assert isinstance(indices, list), "indices must be a list."

        result_files = self.results2img(results, imgfile_prefix, to_label_id, indices)
        return result_files
