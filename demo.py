import os
import aie

from aie.aietorch.datasets.aie.aie_dataset import BinaryChangeDetDataset
from aie.aietorch.trainer.mmseg import ChangeDetTrainer
from aie.aietorch.sampler import RandomNonGeoDatasetSampler



if __name__ == '__main__':
    # 获取aie样本集
    aie.Authenticate()
    aie.g_var.set_var(aie.g_var.GVarKey.Log.LOG_LEVEL, aie.g_var.LogLevel.INFO_LEVEL)
    
    work_dir = "./work_dirs/tutorial"
    
    # 样本集
    myDataSet = BinaryChangeDetDataset(276, data_root=work_dir)
    
    # 随机按照80%， 20%进行切分成两个新样本集
    train_dataset, val_dataset = RandomNonGeoDatasetSampler.split_by_percent(myDataSet, 0.8)
    
    # model
    trainer = ChangeDetTrainer(work_dir=work_dir)
    
    trainer.setup_dataset(train_dataset)
    trainer.setup_dataset(val_dataset, data_type="val")
    
    trainer.train(validate=True)
    
