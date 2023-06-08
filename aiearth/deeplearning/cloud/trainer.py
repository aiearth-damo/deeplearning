import os
from tempfile import NamedTemporaryFile, gettempdir
import inspect
import yaml
import aie
from aie.client import Endpoints
from .aie_client import AIECloudException, AIEClient
from aiearth.deeplearning.utils.file import zip_dir
from aiearth.deeplearning.utils.colormap import generate_color_code_list
from aiearth.deeplearning.job.train_job import TrainJob

DEFAULT_PACKAGE_IGNORE = [
    "*/__pycache__/*",
    "*.pyc"
]

TRAIN_DEFINE_FILE = "train_config.yaml"

class CloudModelAPI():
    api_group = "/trainSDK/api/model"
    host = Endpoints.HOST

    @classmethod
    def __check_success(cls, resp):
        if not resp.json()["success"]:
            raise AIECloudException(resp.json())

    @classmethod
    def __model_create(cls, model_name, model_type, categoryJson, img_std, 
                     img_mean, fp16=False, gpu_num=2 ,desc="", onnx_shape=(1024, 1024)) -> int:
        api_path = "/job"
        url = cls.host + cls.api_group + api_path
        train_config = {
            "train_type": "custom",
            "img_std": img_std,
            "img_mean": img_mean, 
            "fp16": fp16,
            "aie_token": aie.auth.Authenticate.getCurrentUserToken(),
            "gpu_num": gpu_num,
            "onnx_shape": onnx_shape,
        }
        params = {
            "modelType": model_type,
            "modelName": model_name,
            "categoryJson": categoryJson,
            "desc": desc,
            "trainConfig": train_config
        }
        print(params)
        resp = AIEClient.post(url, {}, params)
        cls.__check_success(resp)
        model_id = resp.json()["data"]
        return model_id

    @classmethod    
    def __code_package_upload(cls, model_id: int, file: str):
        api_path = "/upload"
        url = cls.host + cls.api_group + api_path
        data = {
            "model_id": model_id
        }
        files = {'file': open(file, 'rb')}
        ret = AIEClient.post(url, {}, data=data, use_json=False, files=files)
        cls.__check_success(ret)
        print(ret.json())

    @classmethod
    def __model_deploy(cls, model_id: int, onnx_path: str):
        api_path = "/deploy"
        url = cls.host + cls.api_group + api_path
        data = {
            "model_id": model_id
        }
        files = {'file': open(onnx_path, 'rb')}
        ret = AIEClient.post(url, {}, data=data, use_json=False, files=files)
        cls.__check_success(ret)
        print(ret.json())

    @classmethod
    def __code_package(cls, code_dir, package_path) -> str:
        cls.__check_file(code_dir)
        print("starting package %s to %s" % (os.path.realpath(code_dir), package_path))
        zip_dir(package_path, code_dir, code_dir, exclude=DEFAULT_PACKAGE_IGNORE, show_detail=True)
        print("finished package code")
        return package_path

    @classmethod
    def __check_file(cls, code_dir):
        train_config_file = os.path.join(code_dir, TRAIN_DEFINE_FILE)
        if not os.path.exists(train_config_file):
            raise Exception("Must define training config in " + train_config_file)

    @classmethod
    def model_create(cls, code_dir, model_name, model_type, categoryJson, img_std, 
                     img_mean, fp16=False, gpu_num=2 ,desc="", onnx_shape=(1024, 1024)):
        #create temp zip file
        tempdir = gettempdir() + os.sep
        with NamedTemporaryFile(prefix=tempdir, suffix=".zip", delete=True) as fp:
            model_id = cls.__model_create(model_name, model_type, categoryJson, img_std, 
                         img_mean, fp16, gpu_num ,desc, onnx_shape)
            code_pkg_path = fp.name
            cls.__code_package(code_dir, code_pkg_path)
            cls.__code_package_upload(model_id, code_pkg_path)
        return model_id

    @classmethod 
    def model_deploy(cls, onnx_path, model_name, model_type, categoryJson, img_std, 
                     img_mean, fp16=False, gpu_num=2 ,desc="", onnx_shape=(1024, 1024)):
        model_id = cls.__model_create(model_name, model_type, categoryJson, img_std, 
                     img_mean, fp16, gpu_num ,desc, onnx_shape)
        cls.__model_deploy(model_id, onnx_path)
        return model_id
    
    @classmethod
    def model_info(cls, model_id):
        api_path = "/job"
        url = cls.host + cls.api_group + api_path
        data = {
            "modelVersionId": model_id
        }
        ret = AIEClient.get(url, data)
        cls.__check_success(ret)
        model_list = ret.json()["data"]["list"]
        assert len(model_list) == 1
        return model_list[0]


class CloudTrainerWrap:
    def __init__(self, trainer, model_name, code_dir, gpu_num=1, onnx_shape=(1024, 1024), desc="") -> None:
        self.trainer = trainer
        self.code_dir = code_dir
        self.model_name = model_name
        self.gpu_num = gpu_num
        self.desc = desc
        self.onnx_shape = onnx_shape
        self.cloud_model = trainer.to_cloud_model(onnx_shape)
        pass

    def train(self, *args, **kwargs):
        if self.is_cloud_training_env():
            self.trainer.train(*args, **kwargs)
            self.trainer.export_onnx(self.onnx_shape)
        else:
            return self.__submit_job()

    def __submit_job(self):
        classes = self.cloud_model.get_classes()
        category_json = self.__gen_category_json(classes)
        std = self.cloud_model.get_std()
        mean = self.cloud_model.get_mean()
        is_fp16 = self.cloud_model.is_fp16()
        model_id = CloudModelAPI.model_create(
            self.code_dir, 
            model_name=self.model_name, 
            model_type=self.cloud_model.CLOUD_MODEL_TYPE,
            categoryJson=category_json, img_std=std,
            img_mean=mean, fp16=is_fp16, 
            gpu_num=self.gpu_num, desc=self.desc
        )
        return model_id

    def is_cloud_training_env(self) -> bool:
        if "AIE_CLOUD_ENV" in os.environ:
            return True
        return False

    def __gen_category_json(self, classes):
        '''[{"color":"#FF0000","maxValue":1,"text":"变化区域","value":1,"key":1}]'''
        category_json = []
        color_code_list = generate_color_code_list(len(classes))
        idx = 1
        for cls, color in zip(classes, color_code_list):
            category_json.append({
                "color": color,
                "text": cls,
                "value":idx,
            })
        idx += 1
        return category_json
    


class JobCloudWrap:
    def __init__(self, job: TrainJob, model_name, code_dir, gpu_num=1, onnx_shape=(1024, 1024), desc=""):
        self.job = job
        self.cloud_trainer =  CloudTrainerWrap(job.get_trainer(), model_name, code_dir, gpu_num, onnx_shape, desc)
        if not os.path.exists(TRAIN_DEFINE_FILE):
            self.auto_generate_train_config()
    
    def auto_generate_train_config(self):
        print("auto generate train config:", os.path.realpath(TRAIN_DEFINE_FILE))
        job_class = self.job.__class__.__name__
        frame = inspect.stack()[-1]
        module = inspect.getmodule(frame[0])
        filename = module.__file__
        model_name = os.path.relpath(filename, os.getcwd())
        train_config = {
            "entrypoint_script": model_name,
            "job_class": job_class
        }
        with open(TRAIN_DEFINE_FILE, "w") as f:
            yaml.dump(train_config, f)

    def run(self):
        return self.cloud_trainer.train()

        