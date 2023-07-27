import os
from aiearth.deeplearning.datasets.datasets import DatasetType
from mmseg.models import build_segmentor
from aiearth.deeplearning.models.landcover.segmentors import EncoderDecoderLandcover
from aiearth.deeplearning.utils.pytorch2onnx.landcover import to_onnx
from .mmseg_trainer import MMSegTrainer, AIEMMsegCloudModel
from aiearth.deeplearning.models.landcover import build_algorithm
from .mmseg_utils import to_mmseg_config


class AIELandcoverCloudModel(AIEMMsegCloudModel):
    CLOUD_MODEL_TYPE = "LAND_COVER_CLF"


class LandcoverTrainer(MMSegTrainer):
    def __init__(
        self, work_dir="./workspace", config_name="fcn_hr18_1024x1024_16k_landcover"
    ) -> None:
        super(LandcoverTrainer, self).__init__(work_dir=work_dir)
        self.base_cfg = to_mmseg_config("LandCover/" + config_name)

    def setup_datasets(self, dataset, data_type="train"):
        assert dataset.dataset_type.value == DatasetType.single_img.value
        super().setup_datasets(dataset, data_type=data_type)
        for dataset in self.cfg.data[data_type]:
            dataset["classes"] = self.cfg["aie_classes"]

    def to_cloud_model(self, onnx_shape=None) -> AIELandcoverCloudModel:
        return AIELandcoverCloudModel(self.cfg, onnx_shape)

    def export_onnx(self, output_file=None, checkpoint_path=None, shape=(1024, 1024)):
        if not checkpoint_path:
            checkpoint_path = os.path.join(self.work_dir, "latest.pth")
        if not output_file:
            output_file = checkpoint_path.replace(".pth", ".onnx")
        to_onnx(self.cfg, output_file, checkpoint_path, shape)
        return output_file

    def build_model(self):
        algorithm = self.cfg.get("algorithm", None)

        if algorithm is None:
            model = build_segmentor(
                self.cfg.model,
                train_cfg=self.cfg.get("train_cfg"),
                test_cfg=self.cfg.get("test_cfg"),
            )
        else:
            # Different from mmsegmentation
            # replace `model` to `algorithm`
            if self.cfg.algorithm.architecture.model == "same":
                self.cfg.algorithm.architecture.model = self.cfg.model
            model = build_algorithm(self.cfg.algorithm)
        model.init_weights()
        return model
