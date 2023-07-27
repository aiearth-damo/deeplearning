import numpy as np
import cv2
import mmcv
import torch
import torch.nn.functional as F
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.detr import DETR
from ..utils import save_poly_viz


def NonMaxSuppression(x, kernel_size=3, padding=1, stride=1):
    center_idx = kernel_size**2 // 2
    x = x.view(1, 1, 1, -1)

    # Prepare filter
    f = F.unfold(x, kernel_size=3, padding=1, stride=1)
    f = torch.argmax(f, dim=1).unsqueeze(1)
    f = f == center_idx

    return f.squeeze()


def poly2result(polys, labels, poly_scores, point_scores, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if polys.shape[0] == 0:
        return [np.zeros((0, 64, 2), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(polys, torch.Tensor):
            bboxes = polys.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            # '''
            # import pdb
            # pdb.set_trace()
            _polys = []
            for poly, poly_score in zip(polys, poly_scores):
                poly_score = poly_score.sigmoid()
                f01 = poly_score > 0.1  # 0.02
                # f05 = poly_score > 0.4
                # NMS post-process
                # poly2 = poly[f01] if len(poly[f01]) > 2 else []
                nmsf = NonMaxSuppression(poly_score)
                f = f01 & nmsf  # | f05
                poly = poly[f] if len(poly[f]) > 2 else []
                # Ramer-Douglas-Peucker algorithm to simplify polygon
                # if poly2 != []:
                #     rdpf = torch.from_numpy(rdp(poly2.cpu().numpy(), epsilon=30, algo="iter", return_mask=True))
                #     rdpf = rdpf.to(f.device)
                #     rdpf2 = f01
                #     rdpf2[rdpf2==1] = rdpf
                #     f = f | rdpf2 & f05

                # poly = torch.from_numpy(rdp(poly.cpu().numpy(), epsilon=1))
                # poly = poly_rdp.to(poly.device)
                # poly = poly if len(poly) > 2 else []
                # poly = poly[f] if len(poly[f]) > 2 else []
                _polys.append(poly)

        res = [[] for _ in range(num_classes)]
        res[0] = _polys
        # return dict(poly=res, point=point_scores[0] if point_scores is not None else point_scores)
        return res


@DETECTORS.register_module()
class DeformablePoly(DETR):
    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_masks,
        gt_poly_masks=None,
        gt_bboxes_ignore=None,
    ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_masks,
            gt_poly_masks,
            gt_bboxes_ignore,
        )
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_polys, poly_scores, point_scores in results_list
        ]
        poly_results = [
            poly2result(
                det_polys,
                det_labels,
                poly_scores,
                point_scores,
                self.bbox_head.num_classes,
            )
            for det_bboxes, det_labels, det_polys, poly_scores, point_scores in results_list
        ]
        return [(bbox_results[0], poly_results[0])]

    def show_result(
        self,
        img,
        result,
        # gt_polys,
        score_thr=0.3,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        mask_color=None,
        thickness=2,
        font_size=13,
        win_name="",
        show=False,
        wait_time=0,
        out_file=None,
    ):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        img2 = img.copy()
        if isinstance(result, tuple):
            bbox_result, poly_result = result
            if isinstance(poly_result, tuple):
                segm_result = poly_result[0]  # ms rcnn
        else:
            bbox_result, poly_result = result, None
        bbox_result = bbox_result[0]
        if isinstance(poly_result, dict):
            point_result = poly_result["point"]
            poly_result = poly_result["poly"]

        poly_result = poly_result[0]
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        # import pdb
        # pdb.set_trace()
        polygons = []
        for poly, bbox in zip(poly_result, bbox_result):
            if bbox[-1] > score_thr and len(poly) > 0:
                polygons.append(poly.detach().cpu().numpy())
        save_poly_viz(img2, polygons, out_file)

        # cv2.imwrite(out_file[:-4]+'_img.png', img2[:,:,::-1])

        # for i in range(len(gt_polys)):
        #     for j in range(len(gt_polys[i][0])):
        #         gt_polys[i][0][j] = gt_polys[i][0][j].item()
        #     gt_polys[i] = np.array(gt_polys[i]).reshape(-1, 2)
        # save_poly_viz(img2, gt_polys, out_file[:-4]+'_gt.png')

        return polygons
