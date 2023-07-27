import numpy as np
from mmcv.parallel import DataContainer as DC

from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.formatting import (
    DefaultFormatBundle,
    to_tensor,
)


@PIPELINES.register_module()
class DefaultFormatBundleDoubleImage(DefaultFormatBundle):
    def __call__(self, results):
        """Transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if "img1" in results:
            img1 = results["img1"]
            if len(img1.shape) < 3:
                img1 = np.expand_dims(img1, -1)
            img1 = np.ascontiguousarray(img1.transpose(2, 0, 1))
            results["img1"] = DC(to_tensor(img1), stack=True)
        if "img2" in results:
            img2 = results["img2"]
            if len(img2.shape) < 3:
                img2 = np.expand_dims(img2, -1)
            img2 = np.ascontiguousarray(img2.transpose(2, 0, 1))
            results["img2"] = DC(to_tensor(img2), stack=True)
        if "gt_semantic_seg" in results:
            # convert to long
            results["gt_semantic_seg"] = DC(
                to_tensor(results["gt_semantic_seg"][None, ...].astype(np.int64)),
                stack=True,
            )
        if "gt_semantic_seg_sat1" in results:
            # convert to long
            results["gt_semantic_seg_sat1"] = DC(
                to_tensor(results["gt_semantic_seg_sat1"][None, ...].astype(np.int64)),
                stack=True,
            )
        if "gt_semantic_seg_sat2" in results:
            # convert to long
            results["gt_semantic_seg_sat2"] = DC(
                to_tensor(results["gt_semantic_seg_sat2"][None, ...].astype(np.int64)),
                stack=True,
            )
        if "gt_edges" in results:
            # convert to long
            results["gt_edges"] = DC(
                to_tensor(results["gt_edges"][None, ...].astype(np.int64)), stack=True
            )
        if "gt_semantic_seg_multi" in results:
            # convert to long
            results["gt_semantic_seg_multi"] = DC(
                to_tensor(results["gt_semantic_seg_multi"][None, ...].astype(np.int64)),
                stack=True,
            )
        if "shift" in results:
            # convert to long
            results["shift"] = DC(
                to_tensor(results["shift"][None, None, ...].astype(float)), stack=True
            )

        return results
