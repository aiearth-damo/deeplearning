import os
from copy import copy
from .mmseg_trainer import MMSegTrainer,AIEMMsegCloudModel
from .mmseg_utils import to_mmseg_config
from .vars import CONFIG_BASE_DIR

import aiearth.deeplearning.models.changedet as ChangedetModel
from aiearth.deeplearning.datasets.datasets import DatasetType
from aiearth.deeplearning.utils.pytorch2onnx.changedet import to_onnx


class AIEChangeDetCloudModel(AIEMMsegCloudModel):
    CLOUD_MODEL_TYPE = "CHANGE_DETECTION_BIN"

    def get_classes(self):
        # remove backgroup class
        classes = copy(super().get_classes())
        classes = list(classes)
        classes.pop(0)
        return classes

class ChangeDetTrainer(MMSegTrainer):
    config_base_dir = os.path.join(CONFIG_BASE_DIR, "ChangeDet")

    def __init__(self, config_name="effi-b0_base_50k_new256_cosine_lr_batch_128_adamw" ,work_dir="./workspace") -> None:
        super(ChangeDetTrainer, self).__init__(work_dir=work_dir)
        self.base_cfg = to_mmseg_config("ChangeDet/" + config_name)

    def setup_datasets(self, dataset, data_type="train"):
        assert dataset.dataset_type.value == DatasetType.double_img.value
        super().setup_datasets(dataset, data_type=data_type)

    def export_onnx(self, output_file=None, checkpoint_path=None, shape=(1024, 1024)):
        if not checkpoint_path:
            checkpoint_path = os.path.join(self.work_dir, "latest.pth")
        if not output_file:
            output_file = checkpoint_path.replace(".pth", ".onnx")
        to_onnx(self.cfg, output_file, checkpoint_path, shape)
        return output_file
    
    def to_cloud_model(self, onnx_shape=None) -> AIEChangeDetCloudModel:
        return AIEChangeDetCloudModel(self.cfg, onnx_shape)

        
