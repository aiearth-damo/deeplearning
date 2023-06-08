import os
import aie

from aiearth.deeplearning.trainer.mmseg.changedet_trainer import ChangeDetTrainer
from aiearth.deeplearning.sampler import RandomNonGeoDatasetSampler
from aiearth.deeplearning.job.train_job import TrainJob
from aiearth.deeplearning.datasets import ChangeDetNonGeoCustomDataset

class Job(TrainJob):
    work_dir = "./work_dir"
    classes = ['background', 'change']

    def set_trainer(self):
        trainer = ChangeDetTrainer(work_dir=self.work_dir)
        trainer.cfg.runner["max_iters"] = 200    #dict(type='IterBasedRunner', max_iters=20000)
        trainer.cfg.checkpoint_config["interval"]=50
        trainer.cfg.data.samples_per_gpu = 1
        return trainer

    def set_datasets(self):
        myDataSet = ChangeDetNonGeoCustomDataset(
            self.classes,
            (512, 512, 3),
            (512, 512, 1),
            img_dir="datasets/images1",
            img2_dir="datasets/images2", 
            img_suffix=".jpg",
            ann_dir="datasets/annotations", 
            ann_suffix=".png",
        )
        # 随机按照80%， 20%进行切分成两个新样本集
        train_dataset, val_dataset = RandomNonGeoDatasetSampler.split_by_percent(myDataSet, 0.8)
        self.datasets["train"].append(train_dataset)
        self.datasets["val"].append(val_dataset)



if __name__ == '__main__':
    job = Job()
    job.train()

    
