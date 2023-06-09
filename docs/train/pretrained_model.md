# 预训练模型


|  任务类型  | 说明 | 模型 | Crop Size | Uri | config |
| --- | --- | --- | --- | --- | --- |
| 变化检测/ChangeDet  | 通用二分类变化 | hrnet_w18 | 896x896 | aie://ChangeDet/changedet_hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune.pth | [config](../../aiearth/deeplearning/trainer/mmseg/configs/ChangeDet/hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25.py) | 
| 变化检测/BuildingChange | 建筑物变化 | hrnet_w18 | 896x896 | aie://BuildingChange/buildingchange_hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.pth | [config](../../aiearth/deeplearning/trainer/mmseg/configs/BuildingChange/hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.py) |
|  地物分类/LandCover  | 通用地物分类（11类） | hrnet_w48 | | aie://LandCover/landcover_v1.6.pth | [config](../../aiearth/deeplearning/trainer/mmseg/configs/LandCover/fcn_hr48_1024x1024_16k_landcover.py) | 
|  地物分类/LandCover  | 半监督 | hrnet_w48  | | N/A | [config](../../aiearth/deeplearning/trainer/mmseg/configs/LandCover/semi.py) |
| 目标提取/TargetExtraction | 水体提取 | hrnet_w18 | | aie://TargetExtraction/water_fcn_hr18_1024x1024_40k4_bceious1w1.0_semi0108_it1_0108_it2_0103_iter_20000.pth | [config](../../aiearth/deeplearning/trainer/mmseg/configs/TargetExtraction/fcn_hr18_1024x1024_40k4_bceious1w1.0.py) |


## 使用方式


```python
import aie
aie.Authenticate()
aie.Initialize()

# changeDet pretrained model
from aiearth.deeplearning.model_zoo.model import PretrainedModel
model = PretrainedModel("aie://ChangeDet/changedet_hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune.pth")

# 下载到本地的路径
print(model.local_path)

# 用于pretrained
trainer.cfg.model.pretrained = model.local_path

# 用于finetune
trainer.load_from(model)
```

