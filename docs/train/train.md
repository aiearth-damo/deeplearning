# 模型训练文档


## Trainer

Trainer是定义训练配置，训练过程的class。[定义文件](../../aietorch/trainer/trainer.py)

### MMSegTrainer
MMSegTrainer实现了Trainer中定义的接口，可通过mmseg engine发起模型训练。样本集使用参照[云上样本集使用](../dataset/cloud.md)，[本地样本集使用](../dataset/custom.md)

|  算法类型   | 引用地址  | 支持样本集类型
|  ----  | ----  | ---- |
| 变化检测  | from aietorch.trainer.mmseg import ChangeDetTrainer  | aietorch.datasets.aie.BinaryChangeDetDataset, aietorch.datasets.ChangeDetNonGeoCustomDataset |
| 地物分类 | from aietorch.trainer.mmseg.LandcoverTrainer | aietorch.datasets.aie.LandcoverDataset, aietorch.datasets.LandcoverNonGeoCustomDataset | 
| 目标提取（地物识别）| from aietorch.trainer.mmseg.TargetExctractionTrainer | aietorch.dataset.aie.TargetExctractionDataset, aietorch.datasets.TargetExctractionNonGeoCustomDataset,aietorch.datasets.aie.LandcoverDataset |



```python
# 变化检测
ChangeDetTrainer(config_name="effi-b0_base_50k_new256_cosine_lr_batch_128_adamw" ,work_dir="./workspace")

#地物分类
LandcoverTrainer(work_dir="./workspace", config_name="fcn_hr18_1024x1024_16k_landcover")

#目标提取（地物识别）
TargetExtractionTrainer(work_dir="./workspace", config_name="fcn_hr18_1024x1024_40k4_bceious1w1.0")

```

参数说明
* work_dir: 训练任务的工作路径，日志、模型文件会保存在该目录下
* config_name: 预置的配置名称，具体请参考
  + [地物分类](../../aietorch/trainer/mmseg/configs/LandCover/README.md)
  + [地物分类-半监督学习](../../aietorch/trainer/mmseg/configs/LandCover/README.md#配置4-半监督训练mkd)
  + [变化检测](/aietorch/trainer/mmseg/configs/ChangeDet/README.md)
  + [建筑物变化检测](../../aietorch/trainer/mmseg/configs/BuildingChange/README.md)
  + [目标提取](../../aietorch/trainer/mmseg/configs/TargetExtraction/README.md)




## 创建Job
发起模型训练需定义一个模型训练Job类，该类继承自aietorch.job.TrainJob


需要在该类中设定以下两个变量：
|  名称   | 说明  |
|  ----  | ----  |
| classes  | 该模型数据类目列表 |
| work_dir  | 发起训练时的工作目录，训练日志、模型文件均在该目录中保存 |

需要在该类中实现以下几个方法

|  方法   | 说明  |
|  ----  | ----  |
| set_trainer  | 该方法返回一个训练使用的trainer |
| set_datasets  | 该方法中需要生成样本集实例，并按需append至self.datasets["train"]，self.datasets["test"]，self.datasets["val"]中，分别对应训练集，测试集和验证集。



job示例
```python
from aietorch.job import TrainJob
from aietorch.datasets.aie import BinaryChangeDetDataset
from aietorch.trainer.mmseg import ChangeDetTrainer
from aietorch.sampler import RandomNonGeoDatasetSampler

class Job(TrainJob):
    work_dir = "./work_dir"
    classes = ['background', 'changedarea']

    def set_trainer(self):
        trainer = ChangeDetTrainer(work_dir=self.work_dir)
        trainer.cfg.checkpoint_config["interval"]=50
        return trainer

    def set_datasets(self):
        # dataset from AIEarth platform
        myDataSet = BinaryChangeDetDataset(276, data_root=self.work_dir)

        ## dataset from local
        #myDataSet = ChangeDetNonGeoCustomDataset(
        #    self.classes,
        #    (512, 512, 3),
        #    (512, 512, 1),
        #    img_dir="datasets/images1",
        #    img2_dir="datasets/images2", 
        #    img_suffix=".jpg",
        #    ann_dir="datasets/annotations", 
        #    ann_suffix=".png",
        #)
        # 随机按照80%， 20%进行切分成两个新样本集
        train_dataset, val_dataset = RandomNonGeoDatasetSampler.split_by_percent(myDataSet, 0.8)
        self.datasets["train"].append(train_dataset)
        self.datasets["val"].append(val_dataset)

```


## 本地训练


```python
if __name__ == '__main__':
    # 如果需要加载云端数据集，需要设置aie参数
    aie.Authenticate("token")
    aie.g_var.set_var(aie.g_var.GVarKey.Log.LOG_LEVEL, aie.g_var.LogLevel.INFO_LEVEL)
    # 启动本地训练
    job = Job()

    # 启动模型训练job.run() 与job.train()一致
    job.run()
    # 启动测试
    job.test()

    #导出onnx
    job.export_onnx()

```


## 云上训练


使用aietorch.cloud.trainer.JobCloudWrap包装类对本地训练的job进行包装，调用run()方法即可发起云上训练


```python
from aietorch.cloud.trainer import JobCloudWrap

if __name__ == '__main__':
    # 必须设置aie参数
    aie.Authenticate("token")
    aie.g_var.set_var(aie.g_var.GVarKey.Log.LOG_LEVEL, aie.g_var.LogLevel.INFO_LEVEL)
    # 创建任务实例
    job = Job()

    job = JobCloudWrap(
        job=job,
        model_name="changedet_0.1",
        code_dir='.',
        desc="from sdk",
    )

    # 启动云上模型训练
    job.run()

```

JobCloudWrap函数原型

```python
JobCloudWrap(job: TrainJob, model_name, code_dir, gpu_num=1, onnx_shape=(1024, 1024), desc="")
```

参数说明
* job: 训练job实例
* model_name: 模型在aie云平台展示的名称
* code_dir: 本地训练代码路径，该路径下文件会被打包上传至云平台发起云上训练。
* gpu_num: 选择云上训练需要用几张卡，目前最大支持2卡训练
* onnx_shape: 设置导出的onnx支持的shape，默认为1024，1024
* desc: 模型的描述信息，aie平台可以展示

