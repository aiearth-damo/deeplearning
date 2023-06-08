import os
import abc
from enum import Enum

from aiearth.deeplearning.utils.file import list_dir


class DatasetType(Enum):
    single_img = 0
    double_img = 1


class Dataset(metaclass=abc.ABCMeta):

    @abc.abstractproperty
    def img_ann_mapping_list(self) -> list:
        pass

    @property
    def img_dtype(self) -> str:
        pass

    @property
    def ann_dtype(self) -> type:
        pass

    @property
    def datasets_type(self) -> DatasetType:
        pass

    @abc.abstractproperty
    def classes(self) -> list:
        pass

    @abc.abstractproperty
    def img_shape(self) -> tuple:
        pass

    @abc.abstractproperty
    def ann_shape(self) -> tuple:
        pass


class GeoDataset(Dataset):
    pass


class NonGeoDataset(Dataset):
    def __init__(self,
                 classes: tuple or list,
                 img_shape: tuple,
                 ann_shape: tuple,
                 dataset_type: DatasetType,
                 img_dir=None,
                 img2_dir=None,
                 img_suffix=".jpg",
                 ann_dir=None,
                 ann_suffix=".png",
                 img_ann_mapping_list=None,
                 img_dtype="uint8",
                 ann_dtype="uint8",
                 ) -> None:
        self._classes = classes
        self._img_shape = img_shape
        self._ann_shape = ann_shape
        self._img_dir = img_dir
        self._img2_dir = img2_dir
        self._img_suffix = img_suffix
        self._img_dtype = img_dtype
        self._ann_dir = ann_dir
        self._ann_suffix = ann_suffix
        self._ann_dtype = ann_dtype
        self._img_ann_mapping_list = img_ann_mapping_list
        self._dataset_type = dataset_type

        if img_ann_mapping_list == None:
            self._img_ann_mapping_list = self.generate_mapping_list()

    def generate_mapping_list(self):
        assert self._ann_dir != None and self._ann_suffix != None \
            and self._img_dir != None and self._img_suffix != None

        img_list = list_dir(self._img_dir, self._img_suffix, abspath=True)
        #print("img_list:", img_list)
        ann_list = list_dir(self._ann_dir, self._ann_suffix, abspath=True)
        img_ann_mapping_list = []

        if self._dataset_type == DatasetType.double_img:
            assert self._img2_dir != None
            img2_list = list_dir(
                self._img2_dir, self._img_suffix, abspath=True)

        for ann_path in ann_list:
            file_name_without_ext = os.path.splitext(
                os.path.basename(ann_path))[0]
            img_path = os.path.join(os.path.abspath(
                self._img_dir), file_name_without_ext+self._img_suffix)
            if self._dataset_type == DatasetType.single_img:
                if img_path in img_list:
                    img_ann_mapping_list.append({
                        "img": img_path,
                        "ann": ann_path,
                    })
            elif self._dataset_type == DatasetType.double_img:
                img2_path = os.path.join(os.path.abspath(
                    self._img2_dir), file_name_without_ext+self._img_suffix)
                if img_path in img_list and img2_path in img2_list:
                    img_ann_mapping_list.append({
                        "img": img_path,
                        "img2": img2_path,
                        "ann": ann_path,
                    })

        return img_ann_mapping_list

    @property
    def img_ann_mapping_list(self) -> list:
        return self._img_ann_mapping_list

    @img_ann_mapping_list.setter
    def img_ann_mapping_list(self, value) -> list:
        self._img_ann_mapping_list = value

    @property
    def img_dtype(self) -> str:
        return self._img_dtype

    @property
    def ann_dtype(self) -> type:
        return self._ann_dtype

    @property
    def dataset_type(self) -> DatasetType:
        return self._dataset_type

    @property
    def classes(self) -> list:
        return self._classes

    @property
    def img_shape(self) -> tuple:
        return self._img_shape

    @property
    def ann_shape(self) -> tuple:
        return self._ann_shape


class NonGeoCustomDataset(NonGeoDataset):
    pass


class ChangeDetNonGeoCustomDataset(NonGeoCustomDataset):
    def __init__(self, classes: tuple or list,
                 img_shape: tuple,
                 ann_shape: tuple,
                 img_dir=None,
                 img2_dir=None,
                 img_suffix=".jpg",
                 ann_dir=None,
                 ann_suffix=".png",
                 img_ann_mapping_list=None,
                 img_dtype="uint8",
                 ann_dtype="uint8") -> None:
        dataset_type = DatasetType.double_img
        super().__init__(classes, img_shape, ann_shape, dataset_type, img_dir, img2_dir,
                         img_suffix, ann_dir, ann_suffix, img_ann_mapping_list, img_dtype, ann_dtype)
        from .mmseg.changedet import ChangeDetDataset


class LandcoverNonGeoCustomDataset(NonGeoCustomDataset):
    def __init__(self, classes: tuple or list,
                 img_shape: tuple,
                 ann_shape: tuple,
                 img_dir=None,
                 img_suffix=".jpg",
                 ann_dir=None,
                 ann_suffix=".png",
                 img_ann_mapping_list=None,
                 img_dtype="uint8",
                 ann_dtype="uint8") -> None:
        dataset_type = DatasetType.single_img
        img2_dir = None
        super().__init__(classes, img_shape, ann_shape, dataset_type, img_dir, img2_dir,
                         img_suffix, ann_dir, ann_suffix, img_ann_mapping_list, img_dtype, ann_dtype)
        from .mmseg.landcover.landcover import LandcoverLoader


class TargetExtractionNonGeoCustomDataset(NonGeoCustomDataset):
    def __init__(self, classes: tuple or list,
                 img_shape: tuple,
                 ann_shape: tuple,
                 img_dir=None,
                 img_suffix=".jpg",
                 ann_dir=None,
                 ann_suffix=".png",
                 img_ann_mapping_list=None,
                 img_dtype="uint8",
                 ann_dtype="uint8") -> None:
        dataset_type = DatasetType.single_img
        img2_dir = None
        super().__init__(classes, img_shape, ann_shape, dataset_type, img_dir, img2_dir,
                         img_suffix, ann_dir, ann_suffix, img_ann_mapping_list, img_dtype, ann_dtype)
        from .mmseg.target_extraction.remote_sensing import RemoteSensingBinary
