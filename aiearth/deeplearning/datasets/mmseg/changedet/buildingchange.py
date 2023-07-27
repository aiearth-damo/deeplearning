import os.path as osp
import os
import tempfile

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import mmcv
from PIL import Image
import numpy as np
import cv2

Image.MAX_IMAGE_PIXELS = None


@DATASETS.register_module()
class BuildingChangeDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
        binary_label (bool): Whether to use binary label. Default: False.
        ignore_bg (bool): Whether to ignore background. Default: False.
        ignore_labels (list[int]): List of labels to be ignored. Default: [].
        classes (int): Number of classes. Default: None.
    """

    def __init__(
        self,
        split,
        binary_label=False,
        ignore_bg=False,
        ignore_labels=[],
        classes=None,
        **kwargs,
    ):
        self.binary_label = binary_label
        tmp_classes = []
        for i in range(int(classes)):
            tmp_classes.append(str(i))
        self.CLASSES = tmp_classes
        if not isinstance(ignore_labels, list):
            ignore_labels = [ignore_labels]
        self.ignore_bg = ignore_bg
        self.ignore_labels = ignore_labels
        super(BuildingChangeDataset, self).__init__(
            img_suffix=".jpg", seg_map_suffix=".png", split=split, **kwargs
        )

    #         assert osp.exists(self.img_dir) and self.split is not None

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation.

        Args:
            efficient_test (bool): Whether to use efficient test. Default: False.

        Returns:
            list[ndarray]: Ground truth segmentation maps.
        """
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info["ann"]["seg_map"])
            gt_seg_map = mmcv.imread(seg_map, flag="unchanged", backend="pillow")
            if self.binary_label:
                for ignore_label in self.ignore_labels:
                    gt_seg_map[gt_seg_map == ignore_label] = 0
                gt_seg_map[gt_seg_map > 0] = 1
            elif self.ignore_bg:
                gt_seg_map -= 1
            gt_seg_maps.append(gt_seg_map)
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
        record_list = [line.strip().split() for line in open(os.path.join(split))]
        for record in record_list:
            if len(record) == 3:
                img1, img2, seg_map = record
                img1 = osp.join(self.img_dir, img1)
                img2 = osp.join(self.img_dir, img2)
                seg_map = osp.join(self.ann_dir, seg_map)
                img_info = dict(filename=[img1, img2])
                img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            else:
                img1, img2 = record
                img1 = osp.join(self.img_dir, img1)
                img2 = osp.join(self.img_dir, img2)
                img_info = dict(filename=[img1, img2])
                img_infos.append(img_info)
        return img_infos

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
            indices (list[int]): Indices of images to be saved.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if "ann" in self.img_infos[0].keys():
            save_mode = 0
        else:
            save_mode = 1
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for i, idx in enumerate(indices):
            result = results[i]
            if len(result.shape) == 3:
                edge_result = result[1].astype(np.uint8)
                result = result[0]
            else:
                edge_result = None
            if to_label_id:
                result = self._convert_to_label_id(result)
            filenames = self.img_infos[idx]["filename"]
            if save_mode == 0:
                gtname = self.img_infos[idx]["ann"]["seg_map"]
                filename = filenames[0].split("/")[-1]
                basename = osp.splitext(osp.basename(filename))[0]

                png_filename = osp.join(imgfile_prefix, f"{basename}.png")
                cv2.imwrite(png_filename, result)
            elif save_mode == 1:
                filename = filenames[0].split("/")[-1]
                basename = osp.splitext(osp.basename(filename))[0]
                png_filename = osp.join(imgfile_prefix, f"{basename}.png")
                cv2.imwrite(png_filename, result)

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
            indices (list[int]): Indices of images to be saved.

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

    def compute_instance_IoU(self, results, gt_seg_maps, thresh=0.1, min_scale=50):
        """Compute instance IoU.

        Args:
            results (list[ndarray]): Testing results of the dataset.
            gt_seg_maps (list[ndarray]): Ground truth segmentation maps.
            thresh (float): Threshold of IoU. Default: 0.1.
            min_scale (int): Minimum scale of instance. Default: 50.

        Returns:
            tuple: (num_insts, num_correct_insts, num_gt_insts)
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
                assert index not in index_set, error_string(
                    "index({}) is already in index_set({})".format(index, index_set)
                )
                index_set.add(index)
                single_contours = []
                single_contours.append(contours[index])
                child_id = hierarchy[0][index][2]
                while child_id >= 0:
                    assert hierarchy[0][child_id][3] == index
                    assert hierarchy[0][child_id][2] == -1
                    assert child_id not in index_set, error_string(
                        "child_id({}) is already in index_set({})".format(
                            child_id, index_set
                        )
                    )
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
        gt_num = 0
        for result, gt_seg_map in zip(results, gt_seg_maps):
            # TODO merge adjacent instances
            kernel = np.ones((5, 5), np.uint8)
            gt_seg_map = cv2.dilate(gt_seg_map, kernel, iterations=1)
            gt_seg_map = cv2.erode(gt_seg_map, kernel, iterations=1)

            preds_encode = _encode_mask(result, min_scale=min_scale)
            gts_encode = _encode_mask(gt_seg_map, min_scale=min_scale)
            pred_num += len(preds_encode)
            gt_num += len(gts_encode)
            if len(preds_encode) == 0 or len(gts_encode) == 0:
                continue
            iou_matrix = maskUtils.iou(gts_encode, preds_encode, [])
            for i_gt in range(len(gts_encode)):
                if iou_matrix[i_gt, :].max() > thresh:
                    hit_num += 1
                    pred_idx = np.argmax(iou_matrix[i_gt, :])
                    iou_matrix[:, pred_idx] = 0
        recall = hit_num / gt_num
        precision = hit_num / pred_num
        iou = hit_num / (gt_num + pred_num - hit_num)
        return {"InsRecall": recall, "InsPrecision": precision, "InsIoU": iou}

    '''
    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'InsIoU']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        
        ret_metrics = {}

        if 'InsIoU' in metric:
            metric.remove('InsIoU')
            ret_metrics.update(self.compute_instance_IoU(results, gt_seg_maps))

        ret_metrics.update(eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label))

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics.pop('InsRecall', None)
        ret_metrics.pop('InsPrecision', None)
        ret_metrics.pop('InsIoU', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key in ['aAcc', 'InsRecall', 'InsPrecision', 'InsIoU']:
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key in ['aAcc', 'InsRecall', 'InsPrecision', 'InsIoU']:
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
        '''
