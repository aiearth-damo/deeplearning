""" Ref: https://zhuanlan.zhihu.com/p/29393415 
"""
import copy
from collections import defaultdict
import pickle

import cv2
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
# import pycocotools.mask as maskUtils

# pylint:disable=invalid-name, no-member, too-many-locals, line-too-long
class InsSegCOCOEvaluator:
    """COCO Evaluator for Instance Segmentation."""

    def __init__(self, min_area):
        """
        Args:
            min_area (int): The minimum area of the instance.
        """
        self.min_area = min_area
        self.cocoGt = COCO()

    def convert_gt(self, gts):
        """
        Convert binary gts into cocoGt format.
        Args:
            cocoGt: empty COCO object
            gts: a list of binary mask
        Ref: /pycocotools/coco.py createIndex
        """

        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list),defaultdict(list)
        global_id = 100
        category_id = 1
        images = []
        categories = []
        annotations = []
        for image_id, gt in enumerate(gts):
            contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            height, width = gt.shape
            img = {'file_name': None, 'height': height, 'width': width, 'id': image_id}
            imgs[image_id] = img
            images.append(img)
            cat = {'supercategory': 'change', 'id': category_id, 'name': 'change'}
            cats[cat['id']] = cat
            categories.append(cat)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                bbox = cv2.boundingRect(contour)
                bbox = [float(item) for item in bbox]
                contour = contour.squeeze(1).astype('float')
                contour = contour.flatten()
                contour = [list(contour)]
                ann = {'segmentation': contour, 'area': area, 'iscrowd': 0,
                       'image_id': image_id, 'bbox': bbox,
                       'category_id': category_id, 'id': global_id}
                anns[global_id] = ann
                imgToAnns[image_id].append(ann)
                catToImgs[category_id].append(image_id)
                annotations.append(ann)
                global_id += 1
        self.cocoGt.anns = anns
        self.cocoGt.imgToAnns = imgToAnns
        self.cocoGt.catToImgs = catToImgs
        self.cocoGt.imgs = imgs
        self.cocoGt.cats = cats
        self.cocoGt.dataset = {'images': images, 'categories': categories,
                          'annotations': annotations}

    @staticmethod
    def py_cpu_nms(dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def convert_pred(self, preds):
        """Ref: loadRes in pycocotools/coco.py."""

        res = COCO()
        res.dataset['images'] = self.cocoGt.dataset['images']
        res.dataset['categories'] = copy.deepcopy(self.cocoGt.dataset['categories'])

        global_id = 200
        category_id = 1
        anns = []
        for image_id, pred in enumerate(preds):
            candidate_contours = []
            candidate_bboxes = []
            for thr in np.linspace(0.1, 0.9, 9):
                pred_mask = (pred>thr).astype('uint8')
                contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_area:
                        continue
                    x, y, w, h = list(cv2.boundingRect(contour))
                    score = pred[y:y+h, x:x+w].sum() / area
                    candidate_contours.append(contour)
                    candidate_bboxes.append([x, y, x+w, y+h, score])
            candidate_bboxes = np.array(candidate_bboxes)
            if len(candidate_bboxes) == 0:
                # some pred bboxes may all be small ones
                continue
            keeps = self.py_cpu_nms(candidate_bboxes, 0.3)
            for keep_index in keeps:
                x1, y1, x2, y2, score = candidate_bboxes[keep_index]
                contour = candidate_contours[keep_index]
                area = cv2.contourArea(contour)
                contour = contour.squeeze(1).astype('float')
                contour = contour.flatten()
                contour = [list(contour)]
                ann = {'segmentation': contour, 'area': area, 'iscrowd': 0,
                       'image_id': image_id, 'bbox': [x1, y1, x2-x1, y2-y1],
                       'category_id': category_id, 'id': global_id, 'score': score}
                anns.append(ann)
                global_id += 1
        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def get_res(self, preds, gts):
        """
        Get res with preds and gts.
        Args:
            preds: a list of float fg_prob
            gts: a list of binary mask.
        """
        # pickle.dump([preds, gts], open('debug.pkl', 'wb'),
        #             pickle.HIGHEST_PROTOCOL)
        self.cocoGt = COCO()
        self.convert_gt(gts)
        cocoDt = self.convert_pred(preds)
        # print(cocoGt.dataset)
        # print(cocoDt.dataset)

        # iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        iou_thrs = [0.1]
        iou_type = 'segm'
        cocoEval = COCOeval(self.cocoGt, cocoDt, iou_type)
        cocoEval.params.catIds = [1]
        cocoEval.params.imgIds = list(self.cocoGt.imgToAnns.keys())
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        eval_results = {}
        for key, value in coco_metric_names.items():
            val = float(f'{cocoEval.stats[value]:.3f}')
            eval_results[key] = val
        return eval_results


def debug_coco_annotation():
    """Debug coco annotation format."""

    cocoGt = COCO('/home/gongyou.zyq/datasets/coco/annotations/instances_val2017.json')
    for ann in cocoGt.dataset['annotations'][:1]:
        print(ann)
        # {'segmentation': [[510.66, 423.01, 511.72, 420.03, 510.45, 416.0, 510.34, 413.02, 510.77, 410.26, 510.77, 407.5, 510.34, 405.16, 511.51, 402.83, 511.41, 400.49, 510.24, 398.16, 509.39, 397.31, 504.61, 399.22, 502.17, 399.64, 500.89, 401.66, 500.47, 402.08, 499.09, 401.87, 495.79, 401.98, 490.59, 401.77, 488.79, 401.77, 485.39, 398.58, 483.9, 397.31, 481.56, 396.35, 478.48, 395.93, 476.68, 396.03, 475.4, 396.77, 473.92, 398.79, 473.28, 399.96, 473.49, 401.87, 474.56, 403.47, 473.07, 405.59, 473.39, 407.71, 476.68, 409.41, 479.23, 409.73, 481.56, 410.69, 480.4, 411.85, 481.35, 414.93, 479.86, 418.65, 477.32, 420.03, 476.04, 422.58, 479.02, 422.58, 480.29, 423.01, 483.79, 419.93, 486.66, 416.21, 490.06, 415.57, 492.18, 416.85, 491.65, 420.24, 492.82, 422.9, 493.56, 424.39, 496.43, 424.6, 498.02, 423.01, 498.13, 421.31, 497.07, 420.03, 497.07, 415.15, 496.33, 414.51, 501.1, 411.96, 502.06, 411.32, 503.02, 415.04, 503.33, 418.12, 501.1, 420.24, 498.98, 421.63, 500.47, 424.39, 505.03, 423.32, 506.2, 421.31, 507.69, 419.5, 506.31, 423.32, 510.03, 423.01, 510.45, 423.01]], 'area': 702.1057499999998, 'iscrowd': 0, 'image_id': 289343, 'bbox': [473.07, 395.93, 38.65, 28.67], 'category_id': 18, 'id': 1768}

    for img in cocoGt.dataset['images'][:1]:
        print(img)
        # {'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
    for cat in cocoGt.dataset['categories'][:1]:
        print(cat)
        # {'supercategory': 'person', 'id': 1, 'name': 'person'}


def debug_real():
    """Debug real image."""

    evaluator = InsSegCOCOEvaluator(min_area=2)
    [preds, gts] = pickle.load(open('/home/gongyou.zyq/remmseg/debug.pkl', 'rb'))
    print('########## Eval for case 0 #####################')
    cv2.imwrite('pred.png', (preds[0] * 255).astype('uint8'))
    cv2.imwrite('gt.png', (gts[0] * 255).astype('uint8'))
    res = evaluator.get_res([preds[0]], [gts[0]])


def main():
    """Main."""

    # COCO has its own min_area in evaluation, might be 32
    evaluator = InsSegCOCOEvaluator(min_area=2)
    [preds, gts] = pickle.load(open('/home/gongyou.zyq/remmseg/debug.pkl', 'rb'))
    print('########## Eval for case 0 #####################')
    res = evaluator.get_res(preds, gts)

    preds = []
    gts = []
    blank = np.zeros((200, 200), dtype=np.uint8)
    pred = blank.copy().astype('float')
    pred[10:20, 10:20] = 0.5
    pred[110:120, 110:120] = 0.6
    gt = blank.copy()
    gt[10:22, 10:22] = 1
    gts.append(gt)
    preds.append(pred)

    print('########## Eval for case 1 #####################')
    res = evaluator.get_res([pred], [gt])

    blank = np.zeros((200, 200), dtype=np.uint8)
    pred = blank.copy().astype('float')
    pred[10:20, 10:20] = 0.9
    pred[110:120, 110:120] = 0.8
    gt = blank.copy()
    gt[10:22, 10:22] = 1
    gts.append(gt)
    preds.append(pred)

    print('########## Eval for case 2 #####################')
    res = evaluator.get_res([pred], [gt])

    print('########## Eval for case 3 #####################')
    res = evaluator.get_res(preds, gts)
    print(res)


if __name__ == '__main__':
    # debug_coco_annotation()
    # debug_real()
    main()
