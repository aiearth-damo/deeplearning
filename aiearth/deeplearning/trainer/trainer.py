from abc import ABC, abstractmethod
from aiearth.deeplearning.cloud.model import AIEModel


class Trainer(ABC):
    @abstractmethod
    def train(self, validate):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def export_onnx(self, shape):
        pass

    @abstractmethod
    def to_cloud_model(self, onnx_shape) -> AIEModel:
        pass

    @abstractmethod
    def set_classes(self, classes_list):
        pass

    def semi(self):
        pass
