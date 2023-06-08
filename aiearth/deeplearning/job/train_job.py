import abc
import os
from aiearth.deeplearning.trainer.trainer import Trainer
from aiearth.deeplearning.utils.flock import Flock

class TrainJob():
    work_dir = "./work_dir"
    classes = None  #['background', "change"]
    trainer = None
    datasets = {
        'train': [],
        'val':   [],
        'test':  [],
    }
    setup_flag = False

    @abc.abstractmethod
    def set_trainer(self) -> Trainer:
        pass

    def get_trainer(self):
        if self.trainer == None:
            self.trainer = self.set_trainer()
        self.set_classes()
        return self.trainer
    
    def set_classes(self):
        self.trainer.set_classes(self.classes)
    
    @abc.abstractmethod
    def set_datasets(self):
        pass

    def setup(self):
        if self.setup_flag:
            return self.trainer
        trainer = self.get_trainer()
        self.set_datasets()
        for data_type, datasets in self.datasets.items():
            for dataset in datasets:
                trainer.setup_datasets(dataset, data_type=data_type)
        self.setup_flag = True
        self.traine = trainer
        return trainer

    def run(self, *args, **kwargs):
        return self.train(*args, **kwargs)    

    def train(self, *args, **kwargs):
        os.makedirs(self.work_dir, exist_ok=True)
        lock_file = os.path.join(self.work_dir, "data.lock")
        flock = Flock(lock_file)
        flock.lock()
        trainer = self.setup()
        flock.un_lock()
        trainer.train(*args, **kwargs)

    def test(self, *args, **kwargs):
        trainer = self.setup()
        trainer.test(*args, **kwargs)

    def export_onnx(self, *args, **kwargs):
        trainer = self.setup()
        trainer.export_onnx(*args, **kwargs)

