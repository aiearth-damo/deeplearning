# AIE SDK

[English](README-EN.md) | 简体中文

## 简介

AIE SDK是一个用于遥感图像处理的深度学习训练框架。它提供了本地与云端联动的能力，丰富的数据集获取与使用，易用性和可扩展性。

## 安装

```
# for devel
pip install -e .

# build package and install by package
bash scripts/build_pkg.sh
pip install dist/aiearth-deeplearning.tar.gz
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
         https://engine-aiearth.aliyun.com/#/utility/auth-token
个人token: 
```


### 快速开始
请参考[快速开始教程](quickstart.ipynb) 快速开始。我们也提供了更多的教程方便你的学习和使用。

## 文档

对于不同类别的任务，我们提供了专门的文档来进行指导。

### 样本集使用


* [云上样本集](docs/dataset/cloud.md)
* [用户自定义本地样本集](docs/dataset/custom.md)

### 创建模型训练Job

* [模型训练任务建立](docs/train/train.md)


### 标准算法任务
* [地物分类](aiearth/deeplearning/trainer/mmseg/configs/LandCover/README.md)
* [地物分类-半监督学习](aiearth/deeplearning/trainer/mmseg/configs/LandCover/README.md#配置4-半监督训练mkd)
* [变化检测](aiearth/deeplearning/trainer/mmseg/configs/ChangeDet/README.md)
* [建筑物变化检测](aiearth/deeplearning/trainer/mmseg/configs/BuildingChange/README.md)
* [目标提取](aiearth/deeplearning/trainer/mmseg/configs/TargetExtraction/README.md)

### 定制算法任务
我们也支持用户基于不同的Engine中提供的开发范式，进行自己的算法定制。比如，定制模型结构、损失函数等。
* [开发一个自己的模型](docs/model/custom_model.md)

## 遥感类模型库(Model Zoo)

|  任务类型  | 说明 | 模型 | Crop Size | Uri | config |
| --- | --- | --- | --- | --- | --- |
| 变化检测/ChangeDet  | 通用二分类变化 | hrnet_w18 | 896x896 | aie://ChangeDet/changedet_hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune.pth | [config](aiearth/deeplearning/trainer/mmseg/configs/ChangeDet/hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25.py) |
| 变化检测/BuildingChange | 建筑物变化 | hrnet_w18 | 896x896 | aie://BuildingChange/buildingchange_hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.pth | [config](aiearth/deeplearning/trainer/mmseg/configs/BuildingChange/hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.py) |
|  地物分类/LandCover  | 通用地物分类（11类） | hrnet_w48 | | aie://LandCover/landcover_v1.6.pth | [config](aiearth/deeplearning/trainer/mmseg/configs/LandCover/fcn_hr48_1024x1024_16k_landcover.py) |
|  地物分类/LandCover  | 半监督 | hrnet_w48  | | N/A | [config](aiearth/deeplearning/trainer/mmseg/configs/LandCover/semi.py) |
| 目标提取/TargetExtraction | 水体提取 | hrnet_w18 | | aie://TargetExtraction/water_fcn_hr18_1024x1024_40k4_bceious1w1.0_semi0108_it1_0108_it2_0103_iter_20000.pth | [config](aiearth/deeplearning/trainer/mmseg/configs/TargetExtraction/fcn_hr18_1024x1024_40k4_bceious1w1.0.py) |


## 开源许可证

本项目使用 [Apache 2.0 开源许可证](LICENSE). 


## 联系方式

本项目由阿里巴巴达摩院AI Earth团队维护，你可以通过如下方式联系我们：

| 钉钉    | 微信  |邮箱  
| :----------- | :-----------: |:-----------: |
| 钉钉群号: 32152986 | 微信公众号: AI Earth数知地球 |aiearth@service.aliyun.com
| ![钉钉群号](https://img.alicdn.com/imgextra/i2/O1CN01XW3sCk1JlBoQ5tKAd_!!6000000001068-2-tps-159-160.png "钉钉群号") | ![钉钉群号](https://img.alicdn.com/imgextra/i2/O1CN0109JceF1W63CuznFtA_!!6000000002738-2-tps-160-160.png "钉钉群号") |
