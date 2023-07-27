import numpy as np
import random

try:
    import imgaug.augmenters as iaa
except:
    iaa = None
import cv2
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Rand_Augment:
    """
    image_size: 输入图像大小
    target_size: crop后目标图像大小，若target_size与image_size不等，crop强制执行
    Numbers: 选择的数据增强算子数目，从transforms中进行选择，数值小于或等于transforms元素个数
    Magnitude: 数据增强的强度，大于0小于max_Magnitude，与模型复杂程度正相关
    max_Magnitude:数据增强的最大强度
    transforms:选择用于数据增强算子列表
    p:概率值，进行数据增强的概率
    """

    def __init__(
        self,
        image_size=(1024, 1024),
        target_size=(1024, 1024),
        Numbers=3,
        Magnitude=7,
        max_Magnitude=10,
        transforms=None,
        p=1.0,
    ):
        print("warning using random aug v2")
        self.transforms = [
            "contrast_gamma",
            "contrast_linear",
            "brightness",
            "brightness_channel",
            "equalize",
            # 'hsv',
            "invert_channel",
            "blur",
            "noise_gau",
            "noise_pos",
            ####new add
            "channle_shuffle",
            "dropout",
            "coarse_dropout",
            "multiply",
            "salt_pepper",
            "solarize",
            "jpeg_compression",
            # 'rotate',
            # 'cloud',
            # 'flip_v',
            # 'flip_h',
            # 'scale',
            # 'shear',
            # 'translateX',
            # 'translateY'
        ]
        assert (
            image_size[0] > 1
            and image_size[1] > 1
            and target_size[0] > 1
            and target_size[1] > 1
        )
        self.image_size = image_size
        self.target_size = target_size
        self.crop_state = False
        if image_size[0] != target_size[0] or image_size[1] != target_size[1]:
            self.crop_state = True
        self.pad_r_half = [0, 0]
        self.pad_c_half = [0, 0]
        self.crop_beg = [0, 0]

        assert p <= 1 and p >= 0
        self.p = p

        if transforms is None:
            self.transforms_input = self.transforms
        else:
            assert len(transforms) <= len(self.transforms)
            for i in range(0, len(transforms)):
                assert transforms[i] in self.transforms
            self.transforms_input = transforms

        if Numbers is None:
            self.Numbers = min(3, len(self.transforms_input) // 2)
        else:
            self.Numbers = Numbers

        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        assert Magnitude >= 0 and Magnitude <= self.max_Magnitude

        self.Magnitude = Magnitude

        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation
            "contrast_gamma": np.linspace(0.5, 1.5, self.max_Magnitude),
            "contrast_linear": np.linspace(0.5, 2.0, self.max_Magnitude),
            "brightness": np.linspace(-20, 50, self.max_Magnitude),
            "brightness_channel": np.linspace(-20, 50, self.max_Magnitude),
            "equalize": [0] * self.max_Magnitude,
            # "hsv":                np.linspace(-20, 20, self.max_Magnitude),
            "invert_channel": [0] * self.max_Magnitude,
            "blur": np.linspace(1.5, 2.0, self.max_Magnitude),
            "noise_gau": np.linspace(0.03 * 255, 0.08 * 255, self.max_Magnitude),
            "noise_pos": np.linspace(10, 15.0, self.max_Magnitude),
            "channle_shuffle": [0] * self.max_Magnitude,
            "dropout": [0] * self.max_Magnitude,
            "coarse_dropout": [0] * self.max_Magnitude,
            "multiply": [0] * self.max_Magnitude,
            "salt_pepper": [0] * self.max_Magnitude,
            "solarize": [0] * self.max_Magnitude,
            "jpeg_compression": [0] * self.max_Magnitude,
            # "rotate":             np.linspace(-90, 90, self.max_Magnitude),
            # "cloud":              [0] * self.max_Magnitude,
            # "flip_v":             [0] * self.max_Magnitude,
            # "flip_h":             [0] * self.max_Magnitude,
            # "scale":              [0] * self.max_Magnitude,
            # "shear":              np.linspace(-15, 15, self.max_Magnitude),
            # "translateX":         np.linspace(-100, 100, self.max_Magnitude),
            # "translateY":         np.linspace(-100, 100, self.max_Magnitude)
        }

        self.func_init = {
            "contrast_gamma": self.initCtsGamma,
            "contrast_linear": self.initCtsLinear,
            "brightness": self.initTuneBrightness,
            "brightness_channel": self.initTuneBrightnessChannel,
            "equalize": self.initEqualize,
            # "hsv":                self.inithsv,
            "invert_channel": self.initInvertChannel,
            "blur": self.initBlur,
            "noise_gau": self.initNoiseGauss,
            "noise_pos": self.initNoisePoisson,
            "channle_shuffle": self.initChannelShuffle,
            "dropout": self.initDropout,
            "coarse_dropout": self.initCoarseDropout,
            "multiply": self.initMultiply,
            "salt_pepper": self.initSaltAndPepper,
            "solarize": self.initSolarize,
            "jpeg_compression": self.initJpegCompression,
            # "rotate":             self.initRotate,
            # "cloud":              self.initCloud,
            # "flip_v":             self.initFlipVerticle,
            # "flip_h":             self.initFlipHorizontal,
            # "scale":              self.initScale,
            # "shear":              self.initShear,
            # "translateX":         self.initTranslate_x,
            # "translateY":         self.initTranslate_y
        }

        self.func = {
            "contrast_gamma": self.randomCtsGamma,
            "contrast_linear": self.randomCtsLinear,
            "brightness": self.randomTuneBrightness,
            "brightness_channel": self.randomTuneBrightnessChannel,
            "equalize": self.randomEqualize,
            # "hsv":                self.randomhsv,
            "invert_channel": self.randomInvertChannel,
            "blur": self.randomBlur,
            "noise_gau": self.randomNoiseGauss,
            "noise_pos": self.randomNoisePoisson,
            "channle_shuffle": self.randomChannleShuffle,
            "dropout": self.randomDropout,
            "coarse_dropout": self.randomCoarseDropout,
            "multiply": self.randomMultiply,
            "salt_pepper": self.randomSaltAndPeper,
            "solarize": self.randomSolarize,
            "jpeg_compression": self.randomJpegCompression,
            # "rotate":             self.randomRotate,
            # "cloud":              self.randomCloud,
            # "flip_v":             self.randomFlipVerticle,
            # "flip_h":             self.randomFlipHorizontal,
            # "scale":              self.randomScale,
            # "shear":              self.randomShear,
            # "translateX":         self.randomTranslate_x,
            # "translateY":         self.randomTranslate_y
        }

        self.initAllAugmentOP(self.Magnitude)

    def initNoiseGauss(self, magnitude):
        self.tune_noise_gas = iaa.AdditiveGaussianNoise(scale=magnitude)

    def randomNoiseGauss(self, image, mask):
        image = self.tune_noise_gas.augment_image(image)
        return image, mask

    def initNoisePoisson(self, magnitude):
        self.tune_noise_pos = iaa.AdditivePoissonNoise(lam=magnitude)

    def randomNoisePoisson(self, image, mask):
        image = self.tune_noise_pos.augment_image(image)
        return image, mask

    def initTuneBrightness(self, magnitude):
        self.tune_brightness = iaa.Add(magnitude)

    def randomTuneBrightness(self, image, mask):
        image = self.tune_brightness.augment_image(image)
        return image, mask

    def initTuneBrightnessChannel(self, magnitude):
        self.tune_brightness_channel = iaa.Add(magnitude, per_channel=True)

    def randomTuneBrightnessChannel(self, image, mask):
        image = self.tune_brightness_channel.augment_image(image)
        return image, mask

    def initEqualize(self, magnitude):
        self.tune_equalize = iaa.HistogramEqualization()

    def randomEqualize(self, image, mask):
        image = self.tune_equalize.augment_image(image)
        return image, mask

    def initBlur(self, magnitude):
        self.tune_blur = iaa.GaussianBlur(magnitude)

    def randomBlur(self, image, mask):
        image = self.tune_blur.augment_image(image)
        return image, mask

    def inithsv(self, magnitude):
        # self.tune_hsv = iaa.AddToHueAndSaturation(int(magnitude), per_channel=True)
        self.tune_hsv = iaa.AddToHueAndSaturation((-20, 20), per_channel=True)

    def randomhsv(self, image, mask):
        image = self.tune_hsv.augment_image(image)
        return image, mask

    def initCtsGamma(self, magnitude):
        self.tune_contrast_gamma = iaa.GammaContrast(magnitude, per_channel=True)

    def randomCtsGamma(self, image, mask):
        image = self.tune_contrast_gamma.augment_image(image)
        return image, mask

    def initCtsLinear(self, magnitude):
        self.tune_contrast_linear = iaa.LinearContrast(magnitude, per_channel=True)

    def randomCtsLinear(self, image, mask):
        image = self.tune_contrast_linear.augment_image(image)
        return image, mask

    def initFlipHorizontal(self, magnitude):
        self.tune_flip_h = iaa.Flipud(1.0)

    def randomFlipHorizontal(self, image, mask):
        image = self.tune_flip_h.augment_image(image)
        mask = self.tune_flip_h.augment_image(mask)
        return image.copy(), mask.copy()

    def initFlipVerticle(self, magnitude):
        self.tune_flip_v = iaa.Fliplr(1.0)

    def randomFlipVerticle(self, image, mask):
        image = self.tune_flip_v.augment_image(image)
        mask = self.tune_flip_v.augment_image(mask)
        return image.copy(), mask.copy()

    def initCloud(self, magnitude):
        self.tune_cloud = iaa.CloudLayer(
            intensity_mean=250,
            intensity_freq_exponent=-2,
            intensity_coarse_scale=10,
            alpha_min=0.3,
            alpha_multiplier=0.5,
            alpha_size_px_max=8,
            alpha_freq_exponent=-2.2,
            sparsity=0.9,
            density_multiplier=0.8,
        )

    def randomCloud(self, image, mask):
        image = self.tune_cloud.augment_image(image)
        assert len(image) == 0, print(len(image), image)
        image = image[0]
        return image, mask

    def initInvertChannel(self, magnitude):
        pass

    def randomInvertChannel(self, image, mask):
        assert image.shape[2] == 3
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        return image, mask

    def initChannelShuffle(self, magnitude):
        self.channel_shuffle = iaa.ChannelShuffle(1)

    def randomChannleShuffle(self, image, mask):
        image = self.channel_shuffle.augment_image(image)
        return image, mask

    def initDropout(self, magnitude):
        self.dropout = iaa.Dropout((0.2, 0.5), per_channel=True)

    def randomDropout(self, image, mask):
        image = self.dropout.augment_image(image)
        return image, mask

    def initCoarseDropout(self, magnitude):
        self.coarse_dropout = iaa.CoarseDropout(0.2, per_channel=True, size_px=(2, 5))

    def randomCoarseDropout(self, image, mask):
        image = self.coarse_dropout.augment_image(image)
        return image, mask

    def initMultiply(self, magnitude):
        self.multiply = iaa.Multiply((0.25, 2.5), per_channel=True)

    def randomMultiply(self, image, mask):
        image = self.multiply.augment_image(image)
        return image, mask

    def initSaltAndPepper(self, magnitude):
        self.salt_pepper = iaa.SaltAndPepper((0, 0.1), per_channel=True)

    def randomSaltAndPeper(self, image, mask):
        image = self.salt_pepper.augment_image(image)
        return image, mask

    def initSolarize(self, magnitude):
        self.Solarize = iaa.Solarize(per_channel=False)

    def randomSolarize(self, image, mask):
        image = self.Solarize.augment_image(image)
        return image, mask

    def initJpegCompression(self, magnitude):
        self.JpegCompression = iaa.JpegCompression(compression=(85, 95))

    def randomJpegCompression(self, image, mask):
        image = self.JpegCompression.augment_image(image)
        return image, mask

    def initRotate(self, magnitude):
        self.tune_rotate = iaa.Affine(rotate=magnitude, order=0)

    def randomRotate(self, image, mask):
        image = self.tune_rotate.augment_image(image)
        mask = self.tune_rotate.augment_image(mask)
        return image, mask

    def initShear(self, magnitude):
        self.tune_shear = iaa.Affine(shear=magnitude, order=0)

    def randomShear(self, image, mask):
        image = self.tune_shear.augment_image(image)
        mask = self.tune_shear.augment_image(mask)
        return image, mask

    def initTranslate_x(self, magnitude):
        self.tune_translate = iaa.Affine(translate_px={"x": int(magnitude), "y": 0})

    def randomTranslate_x(self, image, mask):
        image = self.tune_translate.augment_image(image)
        mask = self.tune_translate.augment_image(mask)
        return image, mask

    def initTranslate_y(self, magnitude):
        self.tune_translate = iaa.Affine(translate_px={"x": 0, "y": int(magnitude)})

    def randomTranslate_y(self, image, mask):
        image = self.tune_translate.augment_image(image)
        mask = self.tune_translate.augment_image(mask)
        return image, mask

    def initScale(self, magnitude):
        scale = random.uniform(0.5, 2)
        self.tune_scale = iaa.Affine(scale=scale, order=0)

    def randomScale(self, image, mask):
        image = self.tune_scale.augment_image(image)
        mask = self.tune_scale.augment_image(mask)
        return image, mask

    def initCrop(self):
        image_size_r = self.image_size[0]
        image_size_c = self.image_size[1]
        target_size_r = self.target_size[0]
        target_size_c = self.target_size[1]

        pad_r = max(target_size_r - image_size_r, 0)
        pad_c = max(target_size_c - image_size_c, 0)
        self.pad_r_half[0] = int(pad_r / 2)
        self.pad_r_half[1] = pad_r - self.pad_r_half[0]

        self.pad_c_half[0] = int(pad_c / 2)
        self.pad_c_half[1] = pad_c - self.pad_c_half[0]

        self.crop_beg[0] = np.random.randint(max(image_size_r - target_size_r, 0))
        self.crop_beg[1] = np.random.randint(max(image_size_c - target_size_c, 0))

    def randomCrop(self, image):
        if (
            self.pad_c_half[0]
            or self.pad_c_half[1]
            or self.pad_r_half[0]
            or self.pad_r_half[1]
        ):
            image = cv2.copyMakeBorder(
                image,
                top=self.pad_r_half[0],
                bottom=self.pad_r_half[1],
                left=self.pad_c_half[0],
                right=self.pad_c_half[1],
                borderType=cv2.BORDER_REFLECT_101,
            )
        image = image[
            self.crop_beg[0] : self.crop_beg[0] + self.target_size[0],
            self.crop_beg[1] : self.crop_beg[1] + self.target_size[1],
        ]
        return image

    def randomCropList(self, list_image, list_mask):
        for i in range(len(list_image)):
            list_image[i] = self.randomCrop(list_image[i])
        for i in range(len(list_mask)):
            list_mask[i] = self.randomCrop(list_mask[i])
        return list_image, list_mask

    def initAllAugmentOP(self, magnitude):
        assert (
            len(self.func_init)
            == len(self.func)
            == len(self.transforms)
            == len(self.ranges)
        )
        M = min(magnitude, self.max_Magnitude)
        for i in range(0, len(self.transforms)):
            op_name = self.transforms[i]
            operation_init = self.func_init[op_name]
            mag = self.ranges[op_name][M]
            operation_init(mag)

    def randAugment(self):
        sampled_ops = random.sample(self.transforms_input, self.Numbers)
        dict_ops = {}
        for i in range(len(sampled_ops)):
            dict_ops[self.transforms_input.index(sampled_ops[i])] = sampled_ops[i]

        sampled_ops_new = [dict_ops[key] for key in sorted(dict_ops.keys())]
        return sampled_ops_new

    def __call__(self, results):
        if np.random.random() > self.p:
            return results
        else:
            operations = self.randAugment()
            image = results["img"]
            mask = results["gt_semantic_seg"]
            for op_name in operations:
                operation = self.func[op_name]
                image, mask = operation(image, mask)
            results["img"] = image
            results["gt_semantic_seg"] = mask
            return results
