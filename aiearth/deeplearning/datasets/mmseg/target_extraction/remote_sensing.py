import os.path as osp

import mmcv
import numpy as np
from PIL import Image
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class RemoteSensingBinary(CustomDataset):
    """RemoteSensingBinary dataset class.

    Args:
        CustomDataset (CustomDataset): CustomDataset class.
    """

    CLASSES = ("background", "target")
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        """Initialize RemoteSensingBinary dataset class."""
        super(RemoteSensingBinary, self).__init__(reduce_zero_label=False, **kwargs)
        assert osp.exists(self.img_dir), self.img_dir

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline.

        Args:
            results (dict): Results dict for pipeline.
        """
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir
        results["seg_prefix"] = self.ann_dir
        results["img_infos"] = self.img_infos
        if self.custom_classes:
            results["label_map"] = self.label_map

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Convert results to image.

        Args:
            results (list): Results list.
            imgfile_prefix (str): Image file prefix.
            to_label_id (bool): Whether to convert to label id.
            indices (list, optional): Indices list. Defaults to None.

        Returns:
            list: Result files list.
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
        """Format results.

        Args:
            results (list): Results list.
            imgfile_prefix (str): Image file prefix.
            to_label_id (bool, optional): Whether to convert to label id. Defaults to True.
            indices (list, optional): Indices list. Defaults to None.

        Returns:
            list: Result files list.
        """
        if indices is None:
            indices = list(range(len(self)))
        assert isinstance(results, list), "results must be a list."
        assert isinstance(indices, list), "indices must be a list."
        result_files = self.results2img(results, imgfile_prefix, to_label_id, indices)
        return result_files

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotations.

        Args:
            img_dir (str): Image directory.
            img_suffix (str): Image suffix.
            ann_dir (str): Annotation directory.
            seg_map_suffix (str): Segmentation map suffix.
            split (str): Split file.

        Returns:
            list: Image information list.
        """
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    if len(line.split(" ")) == 1:
                        img_name = line.strip()
                        img_info = dict(filename=img_name + img_suffix)
                        if ann_dir is not None:
                            seg_map = img_name + seg_map_suffix
                            img_info["ann"] = dict(seg_map=seg_map)
                        img_infos.append(img_info)
                    elif len(line.split(" ")) == 2:
                        img_path, msk_path = line.strip().split(" ")
                        img_info = dict(filename=img_path)
                        img_info["ann"] = dict(seg_map=msk_path)
                        img_infos.append(img_info)
                    else:
                        print("error image")
                        NotImplementedError()
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos
