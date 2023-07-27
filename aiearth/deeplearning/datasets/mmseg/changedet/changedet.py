# -*- conding: utf-8 -*-
import os.path as osp
import os
import math
from collections import OrderedDict
from functools import reduce
import pickle
import numpy as np
import tempfile

import torch

import cv2
from skimage import io
from PIL import Image
from prettytable import PrettyTable

import mmcv
from mmcv.utils import print_log
from mmseg.core import eval_metrics, intersect_and_union
from mmseg.core.evaluation.metrics import total_area_to_metrics
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from .coco_evaluator import InsSegCOCOEvaluator

Image.MAX_IMAGE_PIXELS = None


def fast_hist(a, b, n):
    """Compute the histogram of a and b.

    Args:
        a (numpy.ndarray): The first input array.
        b (numpy.ndarray): The second input array.
        n (int): The number of classes.

    Returns:
        numpy.ndarray: The histogram of a and b.
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_hist(image, label, num_class):
    """Compute the histogram of image and label.

    Args:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The input label.
        num_class (int): The number of classes.

    Returns:
        numpy.ndarray: The histogram of image and label.
    """
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    """Compute the kappa of hist.

    Args:
        hist (numpy.ndarray): The input histogram.

    Returns:
        float: The kappa of hist.
    """
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


@DATASETS.register_module()
class ChangeDetDataset(CustomDataset):
    """Change Detection dataset.

    Args:
        split (str): Split txt file for Change Detection.
    """

    CLASSES = ("background", "changedarea")

    PALETTE = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]

    def __init__(
        self,
        split,
        binary_label=True,
        dilate_label=False,
        change_classes=None,
        index2change=None,
        ignore_bg=False,
        ignore_labels=[],
        semi_root=None,
        convert_mode=None,
        mcd_class_name=None,
        **kwargs
    ):
        """
        Initialize the ChangeDetDataset class.

        Args:
            split (str): The split of the dataset.
            binary_label (bool, optional): Whether to use binary labels. Defaults to True.
            dilate_label (bool, optional): Whether to dilate the labels. Defaults to False.
            change_classes (list, optional): The list of classes. Defaults to None.
            index2change (dict, optional): The dictionary of indices to classes. Defaults to None.
            ignore_bg (bool, optional): Whether to ignore the background. Defaults to False.
            ignore_labels (list, optional): The list of labels to ignore. Defaults to [].
            semi_root (str, optional): The path to the semi-supervised dataset. Defaults to None.
            convert_mode (str, optional): The mode of conversion. Defaults to None.
            mcd_class_name (str, optional): The name of the MCD classes. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.binary_label = binary_label
        self.CLASSES = change_classes
        if self.CLASSES is None:
            self.CLASSES = ("background", "changedarea")
            if not self.binary_label:
                raise NotImplementedError()
        if not isinstance(ignore_labels, list):
            ignore_labels = [ignore_labels]
        self.ignore_bg = ignore_bg
        self.ignore_labels = ignore_labels
        self.dilate_label = dilate_label
        self.semi_root = semi_root
        self.gt_seg_maps = None

        super(ChangeDetDataset, self).__init__(
            img_suffix=".jpg", seg_map_suffix=".png", split=split, **kwargs
        )

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        get_gt_seg_maps = self.get_gt_seg_maps()

        return get_gt_seg_maps[index]

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        for pred, index in zip(preds, indices):
            multi_seg_map = self.get_gt_seg_map_by_idx(index)
            binary_seg_map = (multi_seg_map > 0).astype("uint8")
            if type(pred) is not dict:
                # CD only needs pred, thus no dict is needed
                binary_seg_pred = pred
            else:
                # MCD needs multiple results and thus need dict.
                binary_seg_pred = pred["seg_pred"].squeeze(0)
            binary_pre_eval_results = intersect_and_union(
                binary_seg_pred,
                binary_seg_map,
                2,
                self.ignore_index,
                self.label_map,
                self.reduce_zero_label,
            )
            # compute INS pre_eval at any time.
            binary_pre_eval_results = list(
                binary_pre_eval_results
            )  # + [ins_pre_eval_results]
            if not self.binary_label:
                pass
            else:
                merge_pre_eval = binary_pre_eval_results
            pre_eval_results.append(merge_pre_eval)

        return pre_eval_results

    def compute_sek(self, pred, multi_seg_map):
        """Compute SeK for MCD eval.

        Args:
            pred (dict): The prediction dictionary.
            multi_seg_map (numpy.ndarray): The multi-segmentation map.

        Returns:
            numpy.ndarray: The histogram of the prediction and multi-segmentation map.
        """
        binary_results = pred["seg_pred"]
        bad_indexes = (1.0 - binary_results).astype("bool")
        pre_res = pred["results1"] + 1  # [0, 8] with bg
        post_res = pred["results2"] + 1  # [0, 8] with bg
        pre_res[bad_indexes] = 0
        post_res[bad_indexes] = 0

        num_basic_class = self.fg_class_num + 1
        hist = np.zeros((num_basic_class, num_basic_class))
        temp = {"gt_semantic_seg": multi_seg_map}
        convert_res = self.converter(temp)
        label_array_1 = convert_res["gt_semantic_seg_sat1"]
        label_array_2 = convert_res["gt_semantic_seg_sat2"]
        label_array_1 += 1
        label_array_2 += 1
        label_array_1[label_array_1 == 256] = 0  # [0, 8] with bg
        label_array_2[label_array_2 == 256] = 0  # [0, 8] with bg
        infer_array_1, infer_array_2 = pre_res, post_res
        hist += get_hist(infer_array_1, label_array_1, num_basic_class)
        hist += get_hist(infer_array_2, label_array_2, num_basic_class)
        # For Multi-GPU accumulation
        hist_flat = torch.tensor(hist.flatten())
        return hist_flat

    def pre_eval_to_metrics(
        self, pre_eval_results, metrics=["mIoU"], nan_to_num=None, beta=1
    ):
        """Modified from mmseg built-in."""

        pre_eval_results = tuple(zip(*pre_eval_results))

        total_area_intersect = sum(pre_eval_results[0])
        total_area_union = sum(pre_eval_results[1])
        total_area_pred_label = sum(pre_eval_results[2])
        total_area_label = sum(pre_eval_results[3])

        ret_metrics = total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            metrics,
            nan_to_num,
            beta,
        )

        return ret_metrics

    def multi_pre_eval_to_metrics(self, multi_pre_eval_results):
        """
        Calculates metrics from multi-pre-evaluation results.

        Args:
            multi_pre_eval_results (tuple): Tuple of multi-pre-evaluation results.

        Returns:
            dict: Dictionary of metrics, including "SenseIoU", "SeK", "SenseScore", "MCD_OA", and "MCD_Fscore".
        """

        multi_pre_eval_results = tuple(zip(*multi_pre_eval_results))
        # NOTE: we remove InsIoU
        total_hist_flat = sum(multi_pre_eval_results[4])
        num_basic_class = self.fg_class_num + 1
        total_hist_flat = total_hist_flat.cpu().numpy()
        hist = total_hist_flat.reshape(num_basic_class, num_basic_class)

        hist_fg = hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
        Score = 0.3 * IoU_mean + 0.7 * Sek
        print("Mean IoU = %.5f" % IoU_mean)
        print("Sek = %.5f" % Sek)
        print("Score = %.5f" % Score)

        # Some new metrics by Bi-SRNet
        acc = np.diag(hist).sum() / hist.sum()
        mcd_precision = np.diag(hist_fg).sum() / (hist.sum() - hist[0, :].sum())
        mcd_recall = np.diag(hist_fg).sum() / (hist.sum() - hist[:, 0].sum())
        mcd_fscore = 2 * mcd_precision * mcd_recall / (mcd_precision + mcd_recall)
        return {
            "SenseIoU": IoU_mean,
            "SeK": Sek,
            "SenseScore": Score,
            "MCD_OA": acc,
            "MCD_Fscore": mcd_fscore,
        }

    def ins_pre_eval_to_metrics(self, pre_eval_results):
        """Calculate metrics for the pre-evaluation results of the Change Detection task.

        Args:
            pre_eval_results (tuple): Tuple of pre-evaluation results. The first 4 elements are built-in for CD, and the 5th element is for InsIOU.

        Returns:
            dict: Dictionary containing the metrics for the pre-evaluation results. The metrics are InsRecall, InsPrecision, and InsIoU.
        """

        ins_pre_eval_results = tuple(zip(*pre_eval_results))
        total_hist_flat = sum(ins_pre_eval_results[4])
        hit_num, pred_hit_num, pred_num, gt_num = total_hist_flat
        old_flag = False
        if old_flag:
            recall = hit_num / gt_num
            precision = hit_num / pred_num
            iou = hit_num / (gt_num + pred_num - hit_num)
            return {"InsRecall": recall, "InsPrecision": precision, "InsIoU": iou}
        else:
            recall = hit_num / gt_num
            precision = pred_hit_num / pred_num
            iou = hit_num / (gt_num + pred_num)
            return {"InsRecall": recall, "InsPrecision": precision, "InsIoU": iou}

    def get_gt_seg_maps(self, efficient_test=False):
        """
        Get ground truth segmentation maps for evaluation.
        This function reads the segmentation maps from the annotation directory, and maps the labels to the corresponding classes.
        If binary_label is set to True, all labels greater than 0 and not equal to 255 will be set to 1.
        If dilate_label is set to True, the labels will be dilated using a 7x7 convolution kernel.
        """
        if self.gt_seg_maps is not None:
            return self.gt_seg_maps

        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info["ann"]["seg_map"])
            gt_seg_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
            if self.binary_label:
                for ignore_label in self.ignore_labels:
                    gt_seg_map[gt_seg_map == ignore_label] = 0
                gt_seg_map[(gt_seg_map > 0) & (gt_seg_map != 255)] = 1
                if self.dilate_label:
                    conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                    gt_seg_map = cv2.dilate(gt_seg_map, conv_kernel)
            gt_seg_maps.append(gt_seg_map)
        self.gt_seg_maps = gt_seg_maps
        return gt_seg_maps

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        img_infos = []
        record_list = [
            line.strip().split() for line in open(os.path.join(self.ann_dir, split))
        ]
        for record in record_list:
            if len(record) == 3:
                img1, img2, seg_map = record
                img_info = dict(
                    filename=[
                        os.path.join(self.img_dir, img1),
                        os.path.join(self.img_dir, img2),
                    ]
                )
                img_info["ann"] = dict(seg_map=os.path.join(self.ann_dir, seg_map))
                img_infos.append(img_info)
            else:
                img1, img2 = record
                img_info = dict(
                    filename=[
                        os.path.join(self.img_dir, img1),
                        os.path.join(self.img_dir, img2),
                    ]
                )
                img_infos.append(img_info)
        if self.semi_root is not None:
            semi_record_list = [
                line.strip().split()
                for line in open(os.path.join(self.semi_root, "lst/semi.txt"))
            ]
            for record in semi_record_list:
                if len(record) == 3:
                    img1, img2, seg_map = record
                    img_info = dict(
                        filename=[
                            os.path.join(self.semi_root, img1),
                            os.path.join(self.semi_root, img2),
                        ]
                    )
                    img_info["ann"] = dict(
                        seg_map=os.path.join(self.semi_root, seg_map)
                    )
                    img_infos.append(img_info)
                else:
                    img1, img2 = record
                    img_info = dict(
                        filename=[
                            os.path.join(self.semi_root, img1),
                            os.path.join(self.semi_root, img2),
                        ]
                    )
                    img_infos.append(img_info)
        self.img_dir = ""
        self.ann_dir = ""
        return img_infos

    def results2img_mcd(self, results, imgfile_prefix, indices):
        """
        Convert the results of a ChangeDetection dataset into an image file.

        Args:
            results (list): List of results from the ChangeDetection dataset.
            imgfile_prefix (str): Prefix of the image file.
            indices (list): List of indices of the results.

        Returns:
            result_files (list): List of image files.
        """
        # TODO: custom label
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for i, idx in enumerate(indices):
            result = results[i]["seg_pred"][0]
            pre_res = results[i]["results1"][0]
            post_res = results[i]["results2"][0]
            if self.train_type == "yuzhi":
                combined_res = pre_res * 8 + post_res + 1
            else:
                combined_res = pre_res * self.fg_class_num + post_res + 1
                combined_res = self.converter.label_mapping[combined_res]
            filenames = self.img_infos[idx]["filename"]
            basename = ".".join(filenames[0].split("/")[-1].split(".")[:-1])
            png_filename = os.path.join(imgfile_prefix, basename + ".png")
            if result.shape[0] == 3:
                semi_result = result[1:].astype(np.uint8)
                res = np.ones(result.shape[-2:], dtype=np.uint8) * 255
                res[semi_result[0] < 1] = 0
                res[semi_result[1] > 0] = 1
                combined_res[res == 255] = 255
            else:
                res = result.astype(np.uint8)
            combined_res[res == 0] = 0
            io.imsave(png_filename, combined_res.astype(np.uint8), check_contrast=False)
            result_files.append(png_filename)

        return result_files

    def results2img(self, results, imgfile_prefix, to_label_id, indices):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mcd_flag = isinstance(results[0], dict)
        if mcd_flag:
            result_files = self.results2img_mcd(results, imgfile_prefix, indices)
            return result_files

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for i, idx in enumerate(indices):
            result = results[i]
            if len(result.shape) == 3:
                semi_result = result[1:].astype(np.uint8)
                result = result[0].astype(np.uint8)
            else:
                semi_result = None
            if to_label_id:
                result = self._convert_to_label_id(result)
            filenames = self.img_infos[idx]["filename"]
            basename = ".".join(filenames[0].split("/")[-1].split(".")[:-1])
            png_filename = os.path.join(imgfile_prefix, basename + ".png")
            if semi_result is None:
                io.imsave(png_filename, result.astype(np.uint8), check_contrast=False)
            else:
                res = np.ones_like(result, dtype=np.uint8) * 255
                res[semi_result[0] < 1] = 0
                res[semi_result[1] > 0] = 1
                io.imsave(png_filename, res.astype(np.uint8), check_contrast=False)
            result_files.append(png_filename)

        return result_files

    def format_results(
        self, results, imgfile_prefix=None, to_label_id=False, indices=None
    ):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), "results must be a list."
        assert isinstance(indices, list), "indices must be a list."

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(
            results, imgfile_prefix, to_label_id, indices=indices
        )

        return result_files

    def compute_COCO_AP(self, fg_probs, gt_seg_maps, min_scale=50):
        """Ref: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        """
        coco_evaluator = InsSegCOCOEvaluator(min_scale)
        res = coco_evaluator.get_res(fg_probs, gt_seg_maps)
        return res

    def compute_instance_IoU(self, results, gt_seg_maps, thresh=0.1, min_scale=50):
        """
        Compute the instance-level Intersection over Union (IoU) of the given results and ground truth segmentation maps.

        Args:
            results (list): List of segmentation maps.
            gt_seg_maps (list): List of ground truth segmentation maps.
            thresh (float): Threshold for IoU calculation.
            min_scale (int): Minimum scale for IoU calculation.

        Returns:
            dict: Dictionary containing the recall, precision, and IoU values.
        """
        num_imgs = len(results)
        assert len(gt_seg_maps) == num_imgs
        import pycocotools.mask as maskUtils

        def _encode_mask(mask, min_scale):
            encode_instances = []
            h, w = mask.shape[:2]

            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )

            index_set = set()
            index = 0
            while index >= 0 and len(contours) > 0:
                assert hierarchy[0][index][3] == -1
                index_set.add(index)
                single_contours = []
                single_contours.append(contours[index])
                child_id = hierarchy[0][index][2]
                while child_id >= 0:
                    assert hierarchy[0][child_id][3] == index
                    assert hierarchy[0][child_id][2] == -1
                    index_set.add(child_id)

                    single_contours.append(contours[child_id])

                    brother_id = hierarchy[0][child_id][0]
                    if brother_id >= 0:
                        assert hierarchy[0][brother_id][1] == child_id
                    child_id = brother_id

                # TODO process to single_contours
                _mask = np.zeros([h, w], dtype=np.uint8)
                _mask = cv2.drawContours(_mask, single_contours, -1, 1, -1)
                if _mask.sum() > min_scale:
                    _mask = np.asfortranarray(_mask)
                    encode_instances.append(maskUtils.encode(_mask))

                brother_id = hierarchy[0][index][0]
                if brother_id >= 0:
                    assert hierarchy[0][brother_id][1] == index
                index = brother_id
            assert len(index_set) == len(contours)

            return encode_instances

        hit_num = 0
        pred_num = 0
        pred_hit_num = 0
        gt_num = 0
        old_flag = False
        for result, gt_seg_map in zip(results, gt_seg_maps):
            # TODO merge adjacent instances
            kernel = np.ones((5, 5), np.uint8)
            gt_seg_map = gt_seg_map.astype("uint8")
            gt_seg_map = cv2.dilate(gt_seg_map, kernel, iterations=1)
            gt_seg_map = cv2.erode(gt_seg_map, kernel, iterations=1)

            result = cv2.dilate(result, kernel, iterations=5)

            preds_encode = _encode_mask(result, min_scale=min_scale)
            gts_encode = _encode_mask(gt_seg_map, min_scale=min_scale)
            pred_num += len(preds_encode)
            gt_num += len(gts_encode)
            if len(preds_encode) == 0 or len(gts_encode) == 0:
                continue
            iou_matrix = maskUtils.iou(gts_encode, preds_encode, [])
            if old_flag:
                for i_gt in range(len(gts_encode)):
                    if iou_matrix[i_gt, :].max() > thresh:
                        hit_num += 1
                        pred_idx = np.argmax(iou_matrix[i_gt, :])
                        iou_matrix[:, pred_idx] = 0
            else:
                pred_hit_set = set()
                for i_gt in range(len(gts_encode)):
                    if iou_matrix[i_gt, :].max() > thresh:
                        hit_num += 1
                        pred_hit_set |= set(np.where(iou_matrix[i_gt, :] > 0)[0])
                pred_hit_num += len(pred_hit_set)
        if old_flag:
            recall = hit_num / gt_num
            precision = hit_num / pred_num
            iou = hit_num / (gt_num + pred_num - hit_num)
        else:
            recall = hit_num / gt_num
            precision = pred_hit_num / pred_num
            iou = hit_num / (gt_num + pred_num)
        return {"InsRecall": recall, "InsPrecision": precision, "InsIoU": iou}

    def compute_instance_IoU_pre(self, results, gt_seg_maps, thresh=0.1, min_scale=50):
        """
        Computes the instance IoU of the given results and ground truth segmentation maps.

        Args:
            results (list): List of results.
            gt_seg_maps (list): List of ground truth segmentation maps.
            thresh (float): Threshold for IoU.
            min_scale (int): Minimum scale for IoU.

        Returns:
            hist_flat (torch.tensor): Tensor containing the hit number, predicted hit number, predicted number, and ground truth number.
        """
        num_imgs = len(results)
        assert len(gt_seg_maps) == num_imgs
        import pycocotools.mask as maskUtils

        def _encode_mask(mask, min_scale):
            encode_instances = []
            h, w = mask.shape[:2]

            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )

            index_set = set()
            index = 0
            while index >= 0 and len(contours) > 0:
                assert hierarchy[0][index][3] == -1
                index_set.add(index)
                single_contours = []
                single_contours.append(contours[index])
                child_id = hierarchy[0][index][2]
                while child_id >= 0:
                    assert hierarchy[0][child_id][3] == index
                    assert hierarchy[0][child_id][2] == -1
                    index_set.add(child_id)

                    single_contours.append(contours[child_id])

                    brother_id = hierarchy[0][child_id][0]
                    if brother_id >= 0:
                        assert hierarchy[0][brother_id][1] == child_id
                    child_id = brother_id

                # TODO process to single_contours
                _mask = np.zeros([h, w], dtype=np.uint8)
                _mask = cv2.drawContours(_mask, single_contours, -1, 1, -1)
                if _mask.sum() > min_scale:
                    _mask = np.asfortranarray(_mask)
                    encode_instances.append(maskUtils.encode(_mask))

                brother_id = hierarchy[0][index][0]
                if brother_id >= 0:
                    assert hierarchy[0][brother_id][1] == index
                index = brother_id
            assert len(index_set) == len(contours)

            return encode_instances

        hit_num = 0
        pred_hit_num = 0
        pred_num = 0
        gt_num = 0
        old_flag = False
        for result, gt_seg_map in zip(results, gt_seg_maps):
            # TODO merge adjacent instances
            kernel = np.ones((5, 5), np.uint8)
            gt_seg_map = gt_seg_map.astype("uint8")
            gt_seg_map = cv2.dilate(gt_seg_map, kernel, iterations=1)
            gt_seg_map = cv2.erode(gt_seg_map, kernel, iterations=1)

            result = cv2.dilate(result, kernel, iterations=5)

            preds_encode = _encode_mask(result, min_scale=min_scale)
            gts_encode = _encode_mask(gt_seg_map, min_scale=min_scale)

            pred_num += len(preds_encode)
            gt_num += len(gts_encode)
            if len(preds_encode) == 0 or len(gts_encode) == 0:
                continue
            # NOTE: issue here
            iou_matrix = maskUtils.iou(gts_encode, preds_encode, [])
            if old_flag:
                for i_gt in range(len(gts_encode)):
                    if iou_matrix[i_gt, :].max() > thresh:
                        hit_num += 1
                        pred_idx = np.argmax(iou_matrix[i_gt, :])
                        iou_matrix[:, pred_idx] = 0
            else:
                pred_hit_set = set()
                for i_gt in range(len(gts_encode)):
                    if iou_matrix[i_gt, :].max() > thresh:
                        hit_num += 1
                        pred_hit_set |= set(np.where(iou_matrix[i_gt, :] > 0)[0])
                pred_hit_num += len(pred_hit_set)
        hist_flat = torch.tensor([hit_num, pred_hit_num, pred_num, gt_num])
        return hist_flat

    def evaluate(
        self, results, metric="mIoU", logger=None, efficient_test=False, **kwargs
    ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            efficient_test (bool): Whether to use efficient test mode.
            **kwargs: Other keyword arguments.

        Returns:
            dict[str, float]: Default metrics.
        """
        # Check if the evaluation is old-style or not
        old_eval_flag = (
            mmcv.is_list_of(results, np.ndarray)
            or mmcv.is_list_of(results, str)
            or mmcv.is_list_of(results, dict)
        )
        if not old_eval_flag:
            # if not mmcv.is_list_of(results, dict):
            # test a list of pre_eval_results
            ret_metrics = {}
            if "InsIoU" in metric:
                ret_metrics.update(self.ins_pre_eval_to_metrics(results))
                metric.remove("InsIoU")
            if "SeK" in metric:
                metric.remove("SeK")
            ret_metrics.update(self.pre_eval_to_metrics(results, metric))
        else:
            print("Old-style without pre_eval")
            if isinstance(metric, str):
                metric = [metric]
            allowed_metrics = ["mIoU", "mDice", "mFscore", "InsIoU", "SeK"]
            if not set(metric).issubset(set(allowed_metrics)):
                raise KeyError("metric {} is not supported".format(metric))
            gt_seg_maps = self.get_gt_seg_maps(efficient_test)
            multi_gt_seg_maps = [item.copy() for item in gt_seg_maps]
            ret_metrics = {}
            seg_preds = results
            if "InsIoU" in metric:
                metric.remove("InsIoU")
                ret_metrics.update(self.compute_instance_IoU(seg_preds, gt_seg_maps))
            if "SeK" in metric:
                metric.remove("SeK")

            ret_metrics.update(
                eval_metrics(
                    seg_preds,
                    gt_seg_maps,
                    2,
                    self.ignore_index,
                    metric,
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                )
            )

            # Clean cache file for old-style without pre_eval
            if mmcv.is_list_of(results, str):
                for file_name in results:
                    os.remove(file_name)

        # Output binary results
        eval_results = {}
        print_log("\n############# Eval for binary class ##############")
        class_names = ("background", "changedarea")
        binary_eval_results = self.output_results(ret_metrics, logger, class_names)
        eval_results.update(binary_eval_results)
        return eval_results

    def compute_sense_metrics(self, multi_pred, multi_gt):
        """Compute sensetime metrics.

        Args:
            multi_pred (list): List of predicted segmentation maps.
            multi_gt (list): List of ground truth segmentation maps.

        Returns:
            dict: Dictionary containing the SenseIoU, SeK, and SenseScore metrics.
        """
        num_basic_class = self.fg_class_num + 1
        hist = np.zeros((num_basic_class, num_basic_class))
        for i, label_array in enumerate(multi_gt):
            label_array_1, label_array_2 = label_array
            infer_array_1, infer_array_2 = multi_pred[i]
            hist += get_hist(infer_array_1, label_array_1, num_basic_class)
            hist += get_hist(infer_array_2, label_array_2, num_basic_class)

        hist_fg = hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
        Score = 0.3 * IoU_mean + 0.7 * Sek
        print("Mean IoU = %.5f" % IoU_mean)
        print("Sek = %.5f" % Sek)
        print("Score = %.5f" % Score)
        return {"SenseIoU": IoU_mean, "SeK": Sek, "SenseScore": Score}

    def output_results(self, ret_metrics, logger, class_names):
        """Output results for CD and MCD.

        This function outputs the results for CD and MCD, including a summary table and a table for each class. It also updates the eval_results dictionary with the metrics and their values.

        Args:
            ret_metrics (dict): A dictionary containing the metrics and their values.
            logger (Logger): A logger object.
            class_names (list): A list of class names.

        Returns:
            eval_results (dict): A dictionary containing the metrics and their values.
        """

        eval_results = {}
        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics.pop("InsRecall", None)
        ret_metrics.pop("InsPrecision", None)
        ret_metrics.pop("InsIoU", None)
        ret_metrics.pop("SenseIoU", None)
        ret_metrics.pop("SeK", None)
        ret_metrics.pop("SenseScore", None)
        ret_metrics.pop("MCD_OA", None)
        ret_metrics.pop("MCD_Fscore", None)
        COCO_AP_METRIC_LIST = [
            "mAP",
            "mAP_50",
            "mAP_75",
            "mAP_s",
            "mAP_m",
            "mAP_l",
            "AR@100",
            "AR@300",
            "AR@1000",
            "AR_s@1000",
            "AR_m@1000",
            "AR_l@1000",
        ]
        for item in COCO_AP_METRIC_LIST:
            ret_metrics.pop(item, None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if (
                key
                in ["aAcc", "InsRecall", "InsPrecision", "InsIoU", "SenseIoU", "SeK"]
                + COCO_AP_METRIC_LIST
            ):
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column("m" + key, [val])

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)
        print_log("Summary:", logger)
        print_log("\n" + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key in ["aAcc", "InsRecall", "InsPrecision", "InsIoU"]:
                eval_results[key] = value / 100.0
            else:
                eval_results["m" + key] = value / 100.0

        ret_metrics_class.pop("Class", None)
        for key, value in ret_metrics_class.items():
            eval_results.update(
                {
                    key + "." + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                }
            )

        return eval_results
