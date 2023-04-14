import abc


class AIEModel:
    @abc.abstractmethod
    def get_model_type(self):
        pass

    @abc.abstractmethod
    def get_meam(self):
        pass

    @abc.abstractmethod
    def get_std(self):
        pass

    @abc.abstractmethod
    def get_classes(self):
        pass
    
    @abc.abstractmethod
    def get_model_name(self):
        pass
    
    @abc.abstractmethod
    def is_fp16(self):
        pass
    
    @abc.abstractmethod
    def get_onnx_shape(self):
        pass
        

    