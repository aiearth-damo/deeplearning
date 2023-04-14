# AIE SDK

[English](README.md) | 简体中文

## 简介

AIE SDK是一个用于遥感图像处理的深度学习框架。它提供了本地与云端联动的能力，丰富的数据集获取与使用，易用性和可扩展性。在训练和推理方面都有很好的表现。

## 最新进展
[2023.3.x]

版本0.1发布。

更多版本的详细信息请参考[变更记录](docs/source/change_log.md)。


## 安装

```
# for devel
pip install -e . -r requirements.txt
mim install mmcv-full==1.7.1
```

## SDK使用

### 前期准备 

SDK与阿里云AI Earth地球科学云平台（简称AIE平台）深度集成，可通过SDK获取云平台的样本集资源，部署模型，提交云上训练等功能，云上功能需要有AIE账号以及生成调用token。

#### 账号注册
账号注册请参考[AIE文档中心](https://engine-aiearth.aliyun.com/docs/page/guide?d=573e72)


#### token申请
在浏览器登录AIE账号后访问[token申请页面](https://engine-aiearth.aliyun.com/#/utility/auth-token)，根据页面提示进行token生成。

在代码中调用aie.Authenticate()，也会提示申请链接并等待输入。效果如下：
```
请将以下地址粘贴到Web浏览器中，访问授权页面，并将个人token粘贴到输入框中
         https://pre-engine-aiearth.aliyun.com/#/utility/auth-token
个人token: 
```


### 快速开始
请参考[快速开始教程](quickstart.ipynb) 快速开始。我们也提供了更多的教程方便你的学习和使用。

## 文档

对于不同类别的任务，我们提供了专门的文档来进行指导。

### 样本集使用


* [云上样本集](docs/dataset/cloud.md)
* [用户自定义本地样本集](docs/dataset/custom.md)



SDK中可直接与AIE云平台模型训练系统中样本集系统关联，无缝调用云平台中公开样本集和个人样本集。可利用aie token下载至本地



### 创建模型训练Job

[模型训练任务建立](/docs/train/train.md)


### 标准算法任务
* [地物分类](aietorch/trainer/mmseg/configs/LandCover/README.md)
* [地物分类-半监督学习](aietorch/trainer/mmseg/configs/LandCover/README.md#配置4-半监督训练mkd)
* [变化检测](aietorch/trainer/mmseg/configs/ChangeDet/README.md)
* [建筑物变化检测](aietorch/trainer/mmseg/configs/BuildingChange/README.md)
* [目标提取](aietorch/trainer/mmseg/configs/TargetExtraction/README.md)

### 定制算法任务
我们也支持用户基于不同的Engine中提供的开发范式，进行自己的算法定制。比如，定制模型结构、损失函数等。
* [开发一个自己的模型](docs/source/custom_model.md)

### 训练上云☁️

## 遥感类模型库(Model Zoo)

|  任务类型  | 说明 | 模型 | Crop Size | Uri | config | 下载 |
| --- | --- | --- | --- | --- | --- | --- | 
| 变化检测/ChangeDet  | 通用二分类变化 | hrnet_w18 | 896x896 | aie://ChangeDet/changedet_hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune.pth | [config](/aietorch/trainer/mmseg/configs/ChangeDet/hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25.py) | [model](https://aie-private-data.oss-cn-hangzhou.aliyuncs.com/model_zoo/ChangeDet/changedet_hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune.pth?OSSAccessKeyId=LTAI5tBdxS7RicYExCHhVv9d&Expires=5279969060&Signature=EMLJ2M1RfuVuzhSF1RhdrEMfDtQ%3D) |
| 变化检测/BuildingChange | 建筑物变化 | hrnet_w18 | 896x896 | aie://BuildingChange/buildingchange_hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.pth | [config](/aietorch/trainer/mmseg/configs/BuildingChange/hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.py) | [model](https://aie-private-data.oss-cn-hangzhou.aliyuncs.com/model_zoo/BuildingChange/buildingchange_hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.pth?OSSAccessKeyId=LTAI5tBdxS7RicYExCHhVv9d&Expires=5279969098&Signature=Sh3YxYEMb6N9JCsI0I6QG594Yl8%3D) |
|  地物分类/LandCover  | 通用地物分类（11类） | hrnet_w48 | | aie://LandCover/landcover_v1.6.pth | [config](/aietorch/trainer/mmseg/configs/LandCover/fcn_hr48_1024x1024_16k_landcover.py) | [model](https://aie-private-data.oss-cn-hangzhou.aliyuncs.com/model_zoo/LandCover/landcover_v1.6.pth?OSSAccessKeyId=LTAI5tBdxS7RicYExCHhVv9d&Expires=1679973126&Signature=N3OtOviGP7v8D%2FEjxYv3hm0nuic%3D) |
|  地物分类/LandCover  | 半监督 | hrnet_w48  | | N/A | [config](/aietorch/trainer/mmseg/configs/LandCover/semi.py) | N/A |
| 目标提取/TargetExtraction | 水体提取 | hrnet_w18 | | aie://TargetExtraction/water_fcn_hr18_1024x1024_40k4_bceious1w1.0_semi0108_it1_0108_it2_0103_iter_20000.pth | [config](/aietorch/trainer/mmseg/configs/TargetExtraction/fcn_hr18_1024x1024_40k4_bceious1w1.0.py) | [model](https://aie-private-data.oss-cn-hangzhou.aliyuncs.com/model_zoo/TargetExtraction/water_fcn_hr18_1024x1024_40k4_bceious1w1.0_semi0108_it1_0108_it2_0103_iter_20000.pth?OSSAccessKeyId=LTAI5tBdxS7RicYExCHhVv9d&Expires=5279969754&Signature=oqmaSbu4XGJ4gL4lcUO%2FAJviEVg%3D) |


## 开源许可证

本项目使用 [Apache 2.0 开源许可证](LICENSE). 项目内含有一些第三方依赖库源码，部分实现借鉴其他开源仓库，仓库名称和开源许可证说明请参考[NOTICE文件](NOTICE)。


## 联系方式

本项目由阿里巴巴达摩院AI Earth团队维护，你可以通过如下方式联系我们：

钉钉群号: 

邮箱: 