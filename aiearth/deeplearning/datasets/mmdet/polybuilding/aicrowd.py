from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class AICrowdDataset(CocoDataset):
    CLASSES = ("building",)
    PALETTE = [(119, 11, 32)]

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        # poly_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            # det, seg, poly = results[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[i])
                    data["score"] = float(bboxes[i][4])
                    data["category_id"] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]

                for i in range(bboxes.shape[0]):
                    if segms[i] is None or segms[i] == []:
                        continue
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[i])
                    data["score"] = float(mask_score[i])
                    data["category_id"] = self.cat_ids[label]
                    if isinstance(segms[i]["counts"], bytes):
                        segms[i]["counts"] = segms[i]["counts"].decode()
                    data["segmentation"] = segms[i]
                    segm_json_results.append(data)

                # poly results
                # some detectors use different scores for bbox and mask
                # polys = poly['poly'][label]
                # mask_score = [bbox[4] for bbox in bboxes]
                # for i in range(bboxes.shape[0]):
                #     if polys[i] == []:
                #         continue
                #     data = dict()
                #     data['image_id'] = img_id
                #     data['bbox'] = self.xyxy2xywh(bboxes[i])
                #     data['score'] = float(mask_score[i])
                #     data['category_id'] = self.cat_ids[label]
                #     if isinstance(polys[i], torch.Tensor):
                #         polys[i] = [polys[i].view(-1).cpu().numpy().tolist()]
                #     data['segmentation'] = polys[i]
                #     poly_json_results.append(data)
        return bbox_json_results, segm_json_results  # , poly_json_results
