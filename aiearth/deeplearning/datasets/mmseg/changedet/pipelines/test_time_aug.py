from mmseg.datasets.pipelines.test_time_aug import MultiScaleFlipAug
from mmseg.datasets.builder import PIPELINES
import mmcv, warnings


@PIPELINES.register_module()
class MultiScaleFlipAugCD(MultiScaleFlipAug):
    """Apply multi-scale and flip augmentations to Change Detection dataset.

    Args:
        transforms (list[dict]): Transforms to apply.
        img_scale (tuple[int]): Image scales.
        img_ratios (tuple[float]): Image aspect ratios.
        flip (bool): Whether to flip images.
        flip_direction (str): Flip direction. Options are "horizontal" and "vertical".
    """

    def __init__(self, **kwargs):
        super(MultiScaleFlipAugCD, self).__init__(**kwargs)

    def __call__(self, results):
        """Call function to apply augmentations to results.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Augmented results.
        """
        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            h, w = results.get("img", results["img1"]).shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio)) for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results["scale"] = scale
                    _results["flip"] = flip
                    _results["flip_direction"] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict
