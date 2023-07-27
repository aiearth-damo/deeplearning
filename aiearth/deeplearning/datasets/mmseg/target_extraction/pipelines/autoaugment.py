# This code is based on https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ImageNetPolicy(object):
    """
    AutoAugment ImageNet Policy
    """

    def __init__(self, fillcolor=(128, 128, 128), fillcolor_mask=(255,), prob=1.0):
        """
        Args:
            fillcolor (tuple): The fill color for image augmentation.
            fillcolor_mask (tuple): The fill color for mask augmentation.
            prob (float): The probability of applying the augmentation.
        """
        self.prob = prob
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor, fillcolor_mask),
            SubPolicy(
                0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor, fillcolor_mask
            ),
            SubPolicy(
                0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor, fillcolor_mask
            ),
            SubPolicy(
                0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor, fillcolor_mask
            ),
            SubPolicy(
                0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor, fillcolor_mask
            ),
            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor, fillcolor_mask),
            SubPolicy(
                0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor, fillcolor_mask
            ),
            SubPolicy(
                0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor, fillcolor_mask
            ),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor, fillcolor_mask),
            SubPolicy(
                0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor, fillcolor_mask
            ),
            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor, fillcolor_mask),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor, fillcolor_mask),
            SubPolicy(
                0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor, fillcolor_mask
            ),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor, fillcolor_mask),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor, fillcolor_mask),
            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor, fillcolor_mask),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor, fillcolor_mask),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor, fillcolor_mask),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor, fillcolor_mask),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor, fillcolor_mask),
            SubPolicy(
                0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor, fillcolor_mask
            ),
            SubPolicy(
                0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor, fillcolor_mask
            ),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor, fillcolor_mask),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor, fillcolor_mask),
            SubPolicy(
                0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor, fillcolor_mask
            ),
        ]

    def __call__(self, results):
        """
        Call function to apply the augmentation.

        Args:
            results (dict): The input data dictionary.

        Returns:
            dict: The augmented data dictionary.
        """
        if np.random.random() > self.prob:
            return results
        policy_idx = random.randint(0, len(self.policies) - 1)
        img, mask = self.policies[policy_idx](
            results["img"], results["gt_semantic_seg"]
        )
        results["img"] = img
        results["gt_semantic_seg"] = mask
        return results

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class SubPolicy(object):
    """
    This class initializes the parameters for the AutoAugment image augmentation technique. It defines the ranges for each operation, the functions for each operation, and the fillcolor and fillcolor_mask.
    """

    def __init__(
        self,
        p1,
        operation1,
        magnitude_idx1,
        p2,
        operation2,
        magnitude_idx2,
        fillcolor=(128, 128, 128),
        fillcolor_mask=(255,),
    ):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        func = {
            "shearX": self._shearX,
            "shearY": self._shearY,
            "translateX": self._translateX,
            "translateY": self._translateY,
            "rotate": self._rotate,
            "color": self._color,
            "posterize": self._posterize,
            "solarize": self._solarize,
            "contrast": self._contrast,
            "sharpness": self._sharpness,
            "brightness": self._brightness,
            "autocontrast": self._autocontrast,
            "equalize": self._equalize,
            "invert": self._invert,
        }
        self.fillcolor = fillcolor
        self.fillcolor_mask = fillcolor_mask
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img, mask):
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if random.random() < self.p1:
            img, mask = self.operation1(img, mask, self.magnitude1)
        if random.random() < self.p2:
            img, mask = self.operation2(img, mask, self.magnitude2)
        img = np.array(img)
        mask = np.array(mask)
        return img, mask

    def _shearX(self, img, mask, magnitude):
        rand = random.choice([-1, 1])
        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, magnitude * rand, 0, 0, 1, 0),
            Image.BILINEAR,
            fillcolor=self.fillcolor,
        )
        mask = mask.transform(
            mask.size,
            Image.AFFINE,
            (1, magnitude * rand, 0, 0, 1, 0),
            Image.NEAREST,
            fillcolor=self.fillcolor_mask,
        )

        return img, mask

    def _shearY(self, img, mask, magnitude):
        rand = random.choice([-1, 1])
        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, magnitude * rand, 1, 0),
            Image.BILINEAR,
            fillcolor=self.fillcolor,
        )
        mask = mask.transform(
            mask.size,
            Image.AFFINE,
            (1, 0, 0, magnitude * rand, 1, 0),
            Image.NEAREST,
            fillcolor=self.fillcolor_mask,
        )
        return img, mask

    def _translateX(self, img, mask, magnitude):
        rand = random.choice([-1, 1])
        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, magnitude * img.size[0] * rand, 0, 1, 0),
            fillcolor=self.fillcolor,
        )
        mask = mask.transform(
            mask.size,
            Image.AFFINE,
            (1, 0, magnitude * mask.size[0] * rand, 0, 1, 0),
            fillcolor=self.fillcolor_mask,
        )
        return img, mask

    def _translateY(self, img, mask, magnitude):
        rand = random.choice([-1, 1])
        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, magnitude * img.size[1] * rand),
            fillcolor=self.fillcolor,
        )
        mask = mask.transform(
            mask.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, magnitude * mask.size[1] * rand),
            fillcolor=self.fillcolor_mask,
        )
        return img, mask

    # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    def _rotate(self, img, mask, magnitude):
        rot = img.convert("RGBA").rotate(magnitude)
        img = Image.composite(
            rot, Image.new("RGBA", rot.size, (self.fillcolor[0],) * 4), rot
        ).convert(img.mode)
        rot_mask = mask.convert("RGBA").rotate(magnitude)
        mask = Image.composite(
            rot_mask,
            Image.new("RGBA", rot_mask.size, (self.fillcolor_mask[0],) * 4),
            rot_mask,
        ).convert(mask.mode)
        return img, mask

    def _color(self, img, mask, magnitude):
        rand = random.choice([-1, 1])
        img = ImageEnhance.Color(img).enhance(1 + magnitude * rand)
        return img, mask

    def _posterize(self, img, mask, magnitude):
        img = ImageOps.posterize(img, magnitude)
        return img, mask

    def _solarize(self, img, mask, magnitude):
        img = ImageOps.solarize(img, magnitude)
        return img, mask

    def _contrast(self, img, mask, magnitude):
        rand = random.choice([-1, 1])
        img = ImageEnhance.Contrast(img).enhance(1 + magnitude * rand)
        return img, mask

    def _sharpness(self, img, mask, magnitude):
        rand = random.choice([-1, 1])
        img = ImageEnhance.Sharpness(img).enhance(1 + magnitude * rand)
        return img, mask

    def _brightness(self, img, mask, magnitude):
        rand = random.choice([-1, 1])
        img = ImageEnhance.Brightness(img).enhance(1 + magnitude * rand)
        return img, mask

    def _autocontrast(self, img, mask, magnitude):
        img = ImageOps.autocontrast(img)
        return img, mask

    def _equalize(self, img, mask, magnitude):
        img = ImageOps.equalize(img)
        return img, mask

    def _invert(self, img, mask, magnitude):
        img = ImageOps.invert(img)
        return img, mask
