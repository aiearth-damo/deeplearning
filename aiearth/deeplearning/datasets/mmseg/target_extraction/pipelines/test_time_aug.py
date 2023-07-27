from mmseg.datasets.pipelines.test_time_aug import MultiScaleFlipAug
from mmseg.datasets.builder import PIPELINES
import mmcv, warnings


@PIPELINES.register_module()
class MultiScaleFlipAugRS(MultiScaleFlipAug):
    def __init__(self, transforms, rotate90=False, rotate_degree=0, **kwargs):
        """
        Args:
            transforms (list[dict]): list of transform configs.
            rotate90 (bool): Whether to rotate the image by 90 degrees.
            rotate_degree (int | list[int]): Degree(s) to rotate the image.
            **kwargs: other keyword arguments.
        """
        super(MultiScaleFlipAugRS, self).__init__(transforms=transforms, **kwargs)
        self.rotate90 = rotate90
        self.rotate_degree = (
            rotate_degree if isinstance(rotate_degree, list) else [rotate_degree]
        )
        assert mmcv.is_list_of(self.rotate_degree, int)
        if not self.rotate90 and self.rotate_degree != [0]:
            warnings.warn("rotate_degree has no effect when rotate is set to False")
        if self.rotate90 and not any([t["type"] == "RandomROT90" for t in transforms]):
            warnings.warn("rotate has no effect when RandomROT90 is not in transforms")
        if not self.rotate90 and all([t["type"] == "RandomROT90" for t in transforms]):
            warnings.warn("not rotate has no effect when RandomROT90 is in transforms")
        if self.rotate90:
            collect = self.transforms.transforms[-1].meta_keys
            assert "rotate90" in collect and "rotate_degree" in collect, collect
        assert 0 in self.rotate_degree, "0 degree must in rotate_degree"

    def __call__(self, results):
        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            h, w = results["img"].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio)) for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            if self.rotate90:
                for deg in self.rotate_degree:
                    for flip in flip_aug:
                        if flip:
                            for direction in self.flip_direction:
                                _results = results.copy()
                                _results["scale"] = scale
                                _results["flip"] = flip
                                _results["flip_direction"] = direction
                                _results["rotate90"] = self.rotate90
                                _results["rotate_degree"] = deg
                                data = self.transforms(_results)
                                aug_data.append(data)
                        else:
                            _results = results.copy()
                            _results["scale"] = scale
                            _results["flip"] = flip
                            _results["flip_direction"] = None
                            _results["rotate90"] = self.rotate90
                            _results["rotate_degree"] = deg
                            data = self.transforms(_results)
                            aug_data.append(data)
            else:
                for flip in flip_aug:
                    if flip:
                        for direction in self.flip_direction:
                            _results = results.copy()
                            _results["scale"] = scale
                            _results["flip"] = flip
                            _results["flip_direction"] = direction
                            _results["rotate90"] = self.rotate90
                            _results["rotate_degree"] = None
                            data = self.transforms(_results)
                            aug_data.append(data)
                    else:
                        _results = results.copy()
                        _results["scale"] = scale
                        _results["flip"] = flip
                        _results["flip_direction"] = None
                        _results["rotate90"] = self.rotate90
                        _results["rotate_degree"] = None
                        data = self.transforms(_results)
                        aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict
