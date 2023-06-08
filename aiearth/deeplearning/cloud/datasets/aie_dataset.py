import json
import os
import re
from aiearth.deeplearning.cloud.aie_client import AIEClient, Endpoints, AIEError, g_var
from aiearth.deeplearning.datasets.datasets import NonGeoDataset, DatasetType
from aiearth.deeplearning.datasets.common.cv_ann_writer import UniversalCVAnnFileWriter
import aiearth.deeplearning.datasets.common.geo_transfer as geo_transfer
import requests
import shutil
import tempfile


class AIEDataSetType:
    # 单景
    __single__ = "single"
    # 双景
    __double__ = "double"


class AIEDataSet(NonGeoDataset, AIEClient):
    __inner_lst_train_dir = "lst/train.txt"
    __segment_split = " "
    __img_suffix = ".jpg"
    __seg_map_suffix = ".png"
    __lst_name = "lst.txt"
    __sampleinfo_host = Endpoints.HOST + "/trainSDK"
    __sampleinfo_api_path = "/api/sdk/sampleset/dinfo?id="

    def __init__(self, dataset_id, data_root=os.getcwd(), classes_filter=[]):
        self.dataset = dataset_id
        self.data_root = data_root
        # download 数据
        self.__get_sampleset(dataset_id)

        # 读取lst数据
        self.img_dir = self.root_dir
        self.ann_dir = self.root_dir

        # 样本集支持选择部分类目训练
        self.classes_filter = classes_filter

        lst_train_file = open(os.path.join(
            self.root_dir, self.__inner_lst_train_dir), 'r', encoding="UTF-8")
        self.inner_lst_train_file = lst_train_file.readlines()
        lst_train_file.close()
        self.__check_type(self.inner_lst_train_file)

        # 预处理样本集
        self._img_ann_mapping_list = []
        self.__preprocess(self.inner_lst_train_file)

    def set_classes_filter(self, classes_filter: list[str]):
        self.classes_filter = classes_filter
        self.__preprocess(self.inner_lst_train_file)

    def restore_classes(self):
        self.classes_filter = self._classes
        self.__preprocess(self.inner_lst_train_file)
        self.classes_filter = []

    def __check_type(self, inner_lst_train_file):
        if (len(inner_lst_train_file) > 0):
            itemStr = inner_lst_train_file[0].split(self.__segment_split)
            if (len(itemStr) > 3):
                self.type = AIEDataSetType.__double__
            else:
                self.type = AIEDataSetType.__single__
        else:
            self.type = AIEDataSetType.__single__

    def __get_sampleset(self, dataset_id):
        print("requelt sampletset info start")
        headers = {"Content-Type": "application/json"}
        get_sampleset_url = self.__sampleinfo_host + \
            self.__sampleinfo_api_path + str(dataset_id)
        print(get_sampleset_url)
        env = os.getenv(g_var.JupyterEnv.TOKEN)
        if env is None:
            get_sampleset_url = '{}&env={}'.format(get_sampleset_url, 'local')
        else:
            get_sampleset_url = '{}&env={}'.format(get_sampleset_url, 'cloud')
        print(get_sampleset_url)

        g_var.set_var(g_var.GVarKey.Log.LOG_LEVEL, g_var.LogLevel.INFO_LEVEL)

        response = super(AIEDataSet, self).get(
            get_sampleset_url, headers).json()
        # print(response)
        if response["code"] != 0:
            raise AIEError(response["code"], response["message"])

        zip_file_url = response["data"]["downloadableLink"]
        self._classes = self.classes_decode(response["data"]["lablemap"])
        self.image_size = self.image_size_decode(response["data"]["cropSize"])
        self._id = response["data"]["id"]
        self._name = response["data"]["name"]
        self._count = response["data"]["count"]
        self._category_md5 = response["data"]["categoryMd5"]

        print("requelt sampletset info end")

        download_path = os.path.join(self.data_root, str(dataset_id))
        if (not os.path.exists(download_path)):
            os.makedirs(download_path)
            self.__download(zip_file_url, download_path)
        else:
            sub_list = os.listdir(download_path)
            if len(sub_list) > 0:
                pass
            else:
                self.__download(zip_file_url, download_path)
        sub_dir = os.listdir(download_path)[0]
        self.root_dir = os.path.join(download_path, sub_dir)
        print("root_dir:" + self.root_dir)
        self.sub_dir_split = sub_dir + "/"

    def classes_decode(self, labelmap: str):
        labelmap = labelmap.strip()
        if labelmap[0] != "[" or labelmap[-1] != "]":
            raise Exception("cannot decode classes:" + labelmap)
        classes = labelmap[1:-1].split(",")
        classes = [cls.strip() for cls in classes]
        return classes

    def image_size_decode(self, image_size: str):
        image_size = image_size.strip()
        ret = [int(i) for i in re.findall(r"(\d+)", image_size)]
        assert len(ret) == 2
        return ret

    def __download(self, zip_file_url, download_path):
        print("download start")
        res = requests.get(url=zip_file_url)
        _tmp_file = tempfile.NamedTemporaryFile()
        _tmp_file.write(res.content)
        shutil.unpack_archive(_tmp_file.name, download_path, format="zip")
        print("download end")

    def __inner_path_split(self, inner_path):
        inner_path_split = inner_path.split(self.sub_dir_split)
        if len(inner_path_split) > 1:
            return inner_path_split[1]
        else:
            return inner_path

    def __get_real_path(self, inner_path):
        return os.path.abspath(os.path.join(self.root_dir, inner_path))

    def __preprocess(self, source_lst):
        print("preprecess start")
        self.img_ann_mapping_list = []
        for i in range(len(source_lst)):
            itemStr = source_lst[i]
            items = itemStr.split(self.__segment_split)
            if (self.type == AIEDataSetType.__single__):
                image_path = self.__inner_path_split(items[0])
                json_path = self.__inner_path_split(items[1])
                ann_path = self.__inner_path_split(
                    str(json_path).replace(".json", self.__seg_map_suffix))
                transform = self.__inner_path_split(items[2].replace("\n", ""))
                self.__geojson_to_png(json_path, transform, ann_path)
                self._img_ann_mapping_list.append({
                    "img": self.__get_real_path(image_path),
                    "ann": self.__get_real_path(ann_path),
                })
            else:
                image_1_path = self.__inner_path_split(items[0])
                image_2_path = self.__inner_path_split(items[1])
                json_path = self.__inner_path_split(items[2])
                ann_path = self.__inner_path_split(
                    str(json_path).replace('.json', self.__seg_map_suffix))
                transform = self.__inner_path_split(items[3].replace("\n", ""))
                self.__geojson_to_png(json_path, transform, ann_path)
                self._img_ann_mapping_list.append({
                    "img":  self.__get_real_path(image_1_path),
                    "img2": self.__get_real_path(image_2_path),
                    "ann":  self.__get_real_path(ann_path),
                })
        #df = pd.DataFrame(lst_file_list, columns=['one'])
        #df.to_csv(target_lst_file_path, columns=['one'], index=False, header=False)
        print("preprecess end")

    def __geojson_to_png(self, json_path, transform, png_path):
        # pass 纯影像数据集不处理
        if (not os.path.exists(self.__get_real_path(json_path))):
            return
        if (not os.path.exists(self.__get_real_path(transform))):
            return
        if (os.path.exists(self.__get_real_path(png_path)) and not self.classes_filter):
            return
        #print(self.__get_real_path(json_path), self.__get_real_path(transform), self.__get_real_path(png_path))
        ann_writer = UniversalCVAnnFileWriter(
            save_path=self.__get_real_path(png_path))
        geo_transfer.trans_annfile_to_image_coordinate(self.__get_real_path(json_path),
                                                       self.__get_real_path(
                                                           transform),
                                                       ann_writer, classes_filter=self.classes_filter)

    @property
    def img_ann_mapping_list(self):
        return self._img_ann_mapping_list

    @img_ann_mapping_list.setter
    def img_ann_mapping_list(self, mapping_list):
        self._img_ann_mapping_list = mapping_list

    @property
    def img_data_type(self) -> str:
        return "uint8"

    @property
    def ann_data_type(self) -> type:
        return "uint8"

    @property
    def img_shape(self) -> tuple[int]:
        return tuple(self.image_size.append(3))

    @property
    def ann_shape(self) -> tuple[int]:
        return tuple(self.image_size.append(1))

    @property
    def classes(self):
        return self._classes if not self.classes_filter else self.classes_filter

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def count(self):
        return self._count

    @property
    def category_md5(self):
        return self._category_md5


class BinaryChangeDetDataset(AIEDataSet):
    def __init__(self, dataset_id, data_root=os.getcwd(), classes_filter=[]):
        super().__init__(dataset_id, data_root, classes_filter)
        from aiearth.deeplearning.datasets.mmseg import ChangeDetDataset

    @property
    def classes(self):
        _classes = super().classes.copy()
        _classes.insert(0, "background")
        return _classes

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.double_img


class LandcoverDataset(AIEDataSet):
    def __init__(self, dataset_id, data_root=os.getcwd(), classes_filter=[]):
        super().__init__(dataset_id, data_root, classes_filter)
        from aiearth.deeplearning.datasets.mmseg.landcover.landcover import LandcoverLoader

    @property
    def classes(self):
        if len(self.classes_filter) != 0:
            _classes = self.classes_filter.copy()
            _classes.insert(0, "background")
            return _classes
        return self._classes

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.single_img


class TargetExtractionDataset(AIEDataSet):
    def __init__(self, dataset_id, data_root=os.getcwd(), classes_filter=[]):
        super().__init__(dataset_id, data_root, classes_filter)
        from aiearth.deeplearning.datasets.mmseg.target_extraction.remote_sensing import RemoteSensingBinary

    @property
    def classes(self):
        _classes = super().classes.copy()
        _classes.insert(0, "background")
        return _classes

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.single_img
