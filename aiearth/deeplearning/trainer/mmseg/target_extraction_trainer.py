from copy import copy
import os

from aiearth.deeplearning.datasets.datasets import DatasetType
from .mmseg_trainer import MMSegTrainer, AIEMMsegCloudModel
from .mmseg_utils import to_mmseg_config
from aiearth.deeplearning.utils.pytorch2onnx.common import to_onnx

class AIETargetExtractionCloudModel(AIEMMsegCloudModel):
    CLOUD_MODEL_TYPE = "TARGET_EXTRACTION"

    def get_classes(self):
        # remove backgroup class
        classes = copy(super().get_classes())
        classes = list(classes)
        classes.pop(0)
        return classes


class TargetExtractionTrainer(MMSegTrainer):
    def __init__(self, work_dir="./workspace", config_name="fcn_hr18_1024x1024_40k4_bceious1w1.0") -> None:
        super(TargetExtractionTrainer, self).__init__(work_dir=work_dir)
        self.base_cfg = to_mmseg_config("TargetExtraction/" + config_name)
        from aiearth.deeplearning.models.target_extraction.segmentors import EncoderDecoderBinary 
        
    def setup_dataset(self, dataset, data_type="train"):
        assert dataset.dataset_type == DatasetType.single_img
        super().setup_datasets(dataset, data_type=data_type)

    def to_cloud_model(self, onnx_shape=None) -> AIETargetExtractionCloudModel:
        return AIETargetExtractionCloudModel(self.cfg, onnx_shape=onnx_shape)
    
    def export_onnx(self, output_file=None, checkpoint_path=None, shape=(1024, 1024)):
        if not checkpoint_path:
            checkpoint_path = os.path.join(self.work_dir, "latest.pth")
        if not output_file:
            output_file = checkpoint_path.replace(".pth", ".onnx")
        to_onnx(self.cfg, output_file, checkpoint_path, shape)
        return output_file

        