import os.path as osp
import mmcv
import numpy as np
from skimage import io
import rasterio
from rasterio.plot import reshape_as_image
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadDoubleImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 bounds_range=None,
                 ):
        """
        Args:
            to_float32 (bool): Whether to convert the loaded image to a float32 numpy array. If set to False, the loaded image is an uint8 array. Defaults to False.
            color_type (str): The flag argument for :func:`mmcv.imfrombytes`. Defaults to 'color'.
            file_client_args (dict): Arguments to instantiate a FileClient. See :class:`mmcv.fileio.FileClient` for details. Defaults to ``dict(backend='disk')``.
            imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default: 'cv2'
            bounds_range (list): The range of the image to be loaded. Defaults to None.
        """
        if bounds_range is None:
            bounds_range = [0, 3]
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.bounds_range = bounds_range

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None and results.get('img_prefix') is not '':
            # filename = osp.join(results['img_prefix'],
            #                     results['img_info']['filename'])
            filename = [osp.join(results['img_prefix'], item) for item in results['img_info']['filename']]
        else:
            filename = results['img_info']['filename']
        for i_img, imgname in enumerate(filename):
            if self.imdecode_backend == 'tif':
                img = io.imread(imgname)
                img = img[:, :, self.bounds_range[0]:self.bounds_range[1]]
            elif imgname.endswith('.img'):
                img = reshape_as_image(rasterio.open(imgname).read()).astype('uint8')
                img = img[:, :, self.bounds_range[0]:self.bounds_range[1]]
            else:
                img_bytes = self.file_client.get(imgname)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if self.to_float32:
                    img = img.astype(np.float32)
            results['img'+str(i_img+1)] = img

        if results['img1'].shape != results['img2'].shape:
            h1, w1 = results['img1'].shape[:2]
            h2, w2 = results['img2'].shape[:2]
            h = min(h1, h2)
            w = min(w1, w2)
            print(f'shape mismatch, img1: {h1},{w1}, img2: {h2},{w2}')
            results['img1'] = results['img1'][:h, :w, :]
            results['img2'] = results['img2'][:h, :w, :]

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        # results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadDoubleAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 fg_class_num,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        """
        Args:
            fg_class_num (int): The number of foreground classes.
            reduce_zero_label (bool): Whether reduce all label value by 1. Usually used for datasets where 0 is background label. Default: False.
            file_client_args (dict): Arguments to instantiate a FileClient. See class:`mmcv.fileio.FileClient` for details. Defaults to ``dict(backend='disk')``.
            imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default: 'pillow'
        """
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.fg_class_num = fg_class_num

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename[0])
        gt_sat1 = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        img_bytes = self.file_client.get(filename[1])
        gt_sat2 = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        gt_semantic_seg = self.combine_map(gt_sat1, gt_sat2)
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def combine_map(self, prev, post):
        """Separate map."""

        # prev, post, [0, self.fg_class_num]
        mask = prev == 0
        prev -= 1    # [-1, self.fg_class_num-1]
        post -= 1
        combined_map = prev * self.fg_class_num + post
        combined_map += 1
        combined_map[mask] = 0    # 0 for bg
        return combined_map

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class Load16bitAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        """
        Args:
            reduce_zero_label (bool): Whether reduce all label value by 1. Usually used for datasets where 0 is background label. Default: False.
            file_client_args (dict): Arguments to instantiate a FileClient. See :class:`mmcv.fileio.FileClient` for details. Defaults to ``dict(backend='disk')``.
            imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default: 'pillow'
        """
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.int64)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results
