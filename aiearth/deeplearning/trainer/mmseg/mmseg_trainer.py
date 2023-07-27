import sys
import os
from copy import copy
import tempfile

from mmseg.datasets.builder import build_dataloader
from ..trainer import Trainer
from ..exception import TrainerException
from .vars import DEFAULT_CFG

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import (
    train_segmentor,
    single_gpu_test,
    multi_gpu_test,
    init_random_seed,
    set_random_seed,
)
from mmseg.utils import (
    build_dp,
    build_ddp,
    collect_env,
    get_device,
    get_root_logger,
    setup_multi_processes,
)
from aiearth.deeplearning.datasets.datasets import NonGeoDataset, DatasetType
from aiearth.deeplearning.cloud.model import AIEModel
from .mmseg_utils import mmseg_base_config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils.config import Config
from torch import nn

import torch


class AIEMMsegCloudModel(AIEModel):
    CLOUD_MODEL_TYPE = None

    def __init__(self, mmseg_config, onnx_shape=None):
        self.mmseg_config = mmseg_config
        self.onnx_shape = None

    def get_mean(self):
        if type(self.mmseg_config.data.train) in (list, tuple):
            train_pipeline = self.mmseg_config.data.train[0]["pipeline"]
        else:
            train_pipeline = self.mmseg_config.data.train["pipeline"]
        for idxpipeline in train_pipeline:
            if (
                "Normalize" == idxpipeline["type"]
                or "DoubleImageNormalize" == idxpipeline["type"]
            ):
                return idxpipeline["mean"]

    def get_std(self):
        if type(self.mmseg_config.data.train) in (list, tuple):
            train_pipeline = self.mmseg_config.data.train[0]["pipeline"]
        else:
            train_pipeline = self.mmseg_config.data.train["pipeline"]
        for idxpipeline in train_pipeline:
            if (
                "Normalize" == idxpipeline["type"]
                or "DoubleImageNormalize" == idxpipeline["type"]
            ):
                return idxpipeline["std"]

    def get_onnx_shape(self):
        if self.onnx_shape != None:
            return self.onnx_shape
        if type(self.mmseg_config.data.train) in (list, tuple):
            train_pipeline = self.mmseg_config.data.train[0]["pipeline"]
        else:
            train_pipeline = self.mmseg_config.data.train["pipeline"]
        for idxpipeline in train_pipeline:
            if (
                "Resize" == idxpipeline["type"]
                or "DoubleImageResize" == idxpipeline["type"]
            ):
                return idxpipeline["img_scale"]
        raise TrainerException("canot find default onnx_shape")

    def get_classes(self):
        return self.mmseg_config.aie_classes

    def get_model_type(self):
        if not self.CLOUD_MODEL_TYPE:
            class_name = self.__class__.__name__
            err_msg = "Class %s is not define CLOUD_MODEL_TYPE" % (class_name)
            raise TrainerException(err_msg)
        return self.CLOUD_MODEL_TYPE

    def is_fp16(self):
        if "Fp16OptimizerHook" in self.mmseg_config.optimizer_config:
            return True
        return False


class MMSegTrainer(Trainer):
    config_base_dir = None

    def __init__(self, work_dir="./workspace") -> None:
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.base_cfg = None
        self.datasets = {
            "train": [],
            "val": [],
            "test": [],
        }

    @property
    def cfg(self):
        if self.base_cfg == None:
            raise TrainerException("base cfg must be init")
        if "work_dir" not in self.base_cfg:
            self.base_cfg.work_dir = self.work_dir
        self.merge_default_value()
        return self.base_cfg

    def merge_config_from_file(self, config_file):
        custom_config = Config.fromfile(config_file)
        self.base_cfg.merge_from_dict(
            custom_config.__getattribute__("_cfg_dict").to_dict()
        )
        print(f"Config:\n{self.cfg.pretty_text}")

    def set_base_cfg_from_file(self, config_file):
        self.base_cfg = mmseg_base_config()
        custom_cfg = Config.fromfile(config_file)
        self.base_cfg.merge_from_dict(
            custom_cfg.__getattribute__("_cfg_dict").to_dict()
        )
        print(f"Config:\n{self.cfg.pretty_text}")

    def build_datasets(self):
        datasets = [build_dataset(self.cfg.data.train)]
        return datasets

    def set_classes(self, classes_list):
        self.cfg["aie_classes"] = classes_list
        return

    def load_from(self, pretrained_model):
        if type(pretrained_model) == str:
            self.cfg.load_from = pretrained_model
        elif hasattr(pretrained_model, "local_path"):
            self.cfg.load_from = pretrained_model.local_path

    def setup_datasets(self, datasets, data_type="train"):
        print(data_type)
        if data_type not in self.datasets:
            raise TrainerException("type must be in %s " % (str(self.datasets.keys())))
        datasets = datasets if type(datasets) in (tuple, list) else [datasets]
        self.datasets[data_type] += datasets
        dataset_cfg = []
        if type(self.cfg.data[data_type]) in (tuple, list):
            data_tmp = Config(self.cfg.data[data_type][0])
        else:
            data_tmp = Config(copy(self.cfg.data[data_type]))

        # set aie_classes
        if "aie_classes" not in self.cfg:
            self.cfg["aie_classes"] = self.datasets[data_type][0].classes

        for dataset in self.datasets[data_type]:
            assert len(dataset.classes) == len(self.cfg.aie_classes)
            data_cfg = self.__setup_nongeodataset(dataset, data_tmp, data_type)
            dataset_cfg.append(data_cfg.__getattribute__("_cfg_dict").to_dict())
        self.cfg.data[data_type] = dataset_cfg

    def __setup_nongeodataset(self, dataset: NonGeoDataset, data_tmpl, data_type=None):
        data_cfg = copy(data_tmpl)
        data_root, split = self.__nongeodataset_to_mmseg_splitfile(dataset, data_type)
        data_cfg.merge_from_dict(
            dict(data_root=data_root, split=split, img_dir="/", ann_dir="/")
        )
        return data_cfg

    def __nongeodataset_to_mmseg_splitfile(
        self, dataset: NonGeoDataset, data_type=None
    ):
        data_root = "/"
        split_lines = []
        if dataset.dataset_type.value == DatasetType.double_img.value:
            for map in dataset.img_ann_mapping_list:
                split_lines.append(" ".join([map["img"], map["img2"], map["ann"]]))
        elif dataset.dataset_type.value == DatasetType.single_img.value:
            for map in dataset.img_ann_mapping_list:
                split_lines.append(" ".join([map["img"], map["ann"]]))
        else:
            raise Exception("dataset type error" + str(dataset.dataset_type))
        split_content = "\n".join(split_lines)
        split_file_prefix = "split_" if data_type == None else data_type + "_split_"
        split_file = tempfile.NamedTemporaryFile(
            "w",
            prefix=split_file_prefix,
            suffix=".txt",
            dir=self.work_dir,
            delete=False,
        )
        split_file.write(split_content)
        split_file.close()
        return data_root, split_file.name

    def build_model(self):
        model = build_segmentor(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg"),
        )
        model.init_weights()
        return model

    def merge_default_value(self):
        for k, v in DEFAULT_CFG.items():
            # trans "a.b.c" = "test" to config["a"]["b"]["c"] = "test"
            sub_keys = k.split(".")
            base_cfg = self.base_cfg
            for sub_key in sub_keys[:-1]:
                if sub_key not in base_cfg:
                    base_cfg[sub_key] = {}
                base_cfg = base_cfg[sub_key]
            if sub_keys[-1] not in base_cfg:
                base_cfg[sub_keys[-1]] = v

    @classmethod
    def list_config(cls):
        files = os.listdir(cls.config_base_dir)
        return [f.strip(".py") for f in files if f.endswith(".py")]

    def get_onnx_script(self):
        package_base_dir = os.path.dirname(sys.modules["train"].__file__)
        return os.path.join(package_base_dir, self.onnx_script)

    def train(self, validate=False, distributed=None):
        if distributed == None:
            distributed = True if "MASTER_ADDR" in os.environ else False
        if distributed:
            init_dist("pytorch", **self.cfg.dist_params)
            # gpu_ids is used to calculate iter when resuming checkpoint
            _, world_size = get_dist_info()
            self.cfg.gpu_ids = range(world_size)
            print(
                "|| MASTER_ADDR:",
                os.environ["MASTER_ADDR"],
                "|| MASTER_PORT:",
                os.environ["MASTER_PORT"],
                "|| LOCAL_RANK:",
                os.environ["LOCAL_RANK"],
                "|| RANK:",
                os.environ["RANK"],
                "|| WORLD_SIZE:",
                os.environ["WORLD_SIZE"],
            )
            print()
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"

        # set multi-process settings
        setup_multi_processes(self.cfg)
        # Build the dataset
        datasets = self.build_datasets()
        # Build the detector
        model = self.build_model()
        # Add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        if "aie_classes" in self.cfg:
            model.CLASSES = self.cfg.aie_classes
        print("cuda available:", torch.cuda.is_available())
        # set random seeds
        self.cfg.device = get_device()
        if not torch.cuda.is_available():
            model = revert_sync_batchnorm(model)
            self.cfg.device = "cpu"
        if not distributed:
            model = revert_sync_batchnorm(model)
        # set random seeds
        seed = init_random_seed(0, device=self.cfg.device)
        seed = seed
        set_random_seed(seed)
        self.cfg.seed = seed
        meta = {}
        meta["seed"] = seed
        meta["exp_name"] = "train"

        print(f"Config:\n{self.cfg.pretty_text}")

        print("classes", model.CLASSES)
        return train_segmentor(
            model,
            datasets,
            self.cfg,
            distributed=distributed,
            validate=validate,
            meta=meta,
        )

    def test(self, checkpoint, output_dir=None, eval=False, metrics=[]):
        dataset = build_dataset(self.cfg.data.test)
        # The default loader config
        loader_cfg = dict(
            # cfg.gpus will be ignored if distributed
            num_gpus=len(self.cfg.gpu_ids),
            dist=False,
            shuffle=False,
        )
        # The overall dataloader settings
        loader_cfg.update(
            {
                k: v
                for k, v in self.cfg.data.items()
                if k
                not in [
                    "train",
                    "val",
                    "test",
                    "train_dataloader",
                    "val_dataloader",
                    "test_dataloader",
                ]
            }
        )
        test_loader_cfg = {
            **loader_cfg,
            "samples_per_gpu": 1,
            "shuffle": False,  # Not shuffle by default
            **self.cfg.data.get("test_dataloader", {}),
        }
        # build the dataloader
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        # build the model and load checkpoint
        self.cfg.model.train_cfg = None
        self.cfg.model.pretrained = None

        model = build_segmentor(self.cfg.model, test_cfg=self.cfg.get("test_cfg"))
        fp16_cfg = self.cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")

        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            print('"CLASSES" not found in meta, use dataset.CLASSES instead')
            model.CLASSES = dataset.CLASSES
        if "PALETTE" in checkpoint.get("meta", {}):
            model.PALETTE = checkpoint["meta"]["PALETTE"]
        else:
            print('"PALETTE" not found in meta, use dataset.PALETTE instead')
            model.PALETTE = dataset.PALETTE

        model = revert_sync_batchnorm(model)
        if not torch.cuda.is_available():
            self.cfg.device = "cpu"
        else:
            # clean gpu memory when starting a new evaluation.
            torch.cuda.empty_cache()

        model = build_dp(model, self.cfg.device, device_ids=self.cfg.gpu_ids)
        # TODO: support multiple images per gpu (only minor changes are needed)
        print(f"Config:\n{self.cfg.pretty_text}")
        # --format-only --eval-options imgfile_prefix=eval_images_save_path
        format_only = False
        eval_kwargs = {}
        if output_dir:
            format_only = True
            eval_kwargs.update(imgfile_prefix=output_dir)
        results = single_gpu_test(
            model,
            data_loader,
            False,
            None,
            False,
            pre_eval=eval,
            format_only=format_only,
            format_args=eval_kwargs,
        )
        if eval and len(metrics) != 0:
            eval_kwargs.update(metric=metrics)
            metrics = dataset.evaluate(results, **eval_kwargs)
            print(metrics)
