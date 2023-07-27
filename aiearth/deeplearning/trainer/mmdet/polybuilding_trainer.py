from copy import copy
import os
import torch

from aiearth.deeplearning.datasets.datasets import DatasetType
from aiearth.deeplearning.utils.pytorch2onnx.common import to_onnx

from .mmdet_trainer import MMDetTrainer, AIEMMdetCloudModel
from .polybuilding_utils import (
    single_gpu_test,
    train_detector,
)
from .mmdet_utils import to_mmdet_config

from mmdet.apis import (
    init_random_seed,
    set_random_seed,
)
from mmdet.utils import (
    get_device,
    setup_multi_processes,
    build_dp,
)
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from mmcv.runner import (
    get_dist_info,
    load_checkpoint,
    wrap_fp16_model,
    get_dist_info,
    init_dist,
)
from mmcv.cnn.utils import revert_sync_batchnorm


class AIEPolyBuildingCloudModel(AIEMMdetCloudModel):
    CLOUD_MODEL_TYPE = "POLY_BUILDING"

    def get_classes(self):
        # remove backgroup class
        classes = copy(super().get_classes())
        classes = list(classes)
        classes.pop(0)
        return classes


class PolyBuildingTrainer(MMDetTrainer):
    def __init__(self, work_dir="./workspace", config_name="default") -> None:
        super(PolyBuildingTrainer, self).__init__(work_dir=work_dir)
        self.base_cfg = to_mmdet_config("PolyBuilding/" + config_name)
        from aiearth.deeplearning.models.polybuilding.detectors import DeformablePoly

    def setup_dataset(self, dataset, data_type="train"):
        assert dataset.dataset_type == DatasetType.single_img
        super().setup_datasets(dataset, data_type=data_type)

    def to_cloud_model(self, onnx_shape=None) -> AIEPolyBuildingCloudModel:
        print(
            "Fire up to CLOUD for this task is NOT PROVIDED now, you can try it locally"
        )
        return AIEPolyBuildingCloudModel(self.cfg, onnx_shape=onnx_shape)

    def export_onnx(self, output_file=None, checkpoint_path=None, shape=(1024, 1024)):
        if not checkpoint_path:
            checkpoint_path = os.path.join(self.work_dir, "latest.pth")
        if not output_file:
            output_file = checkpoint_path.replace(".pth", ".onnx")
        # to_onnx(self.cfg, output_file, checkpoint_path, shape)
        print("Export to ONNX for this task is NOT PROVIDED now")
        return output_file

    def train(self, validate=False, distributed=None):
        if distributed is not None:
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
        return train_detector(
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

        model = build_detector(self.cfg.model, test_cfg=self.cfg.get("test_cfg"))
        fp16_cfg = self.cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")

        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            print('"CLASSES" not found in meta, use dataset.CLASSES instead')
            model.CLASSES = dataset.CLASSES

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
            format_only,  # no show
            output_dir,  # out_dir
        )
        if eval and len(metrics) != 0:
            eval_kwargs.update(metric=metrics)
            metrics = dataset.evaluate(results, **eval_kwargs)
            print(metrics)
