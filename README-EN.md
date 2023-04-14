# AIE SDK

[English](README.md) | [简体中文](README-CN.md)

## Introduction

AIE SDK is a deep learning framework for remote sensing image processing. It provides local and cloud-based capabilities, rich data set acquisition, ease of use, and scalability. It performs well in both training and inference.

## Latest Progress
[2023.3.x]

Version 0.1 released.

For more detailed information on other versions, please refer to the [change log](docs/source/change_log.md).


## Installation

```
# for devel
pip install -e . -r requirements.txt
mim install mmcv-full==1.7.1
```

## SDK Usage

### Preparation 

The SDK is deeply integrated with the Alibaba Cloud AI Earth platform (referred to as the AIE platform), and can obtain sample set resources from the cloud platform, deploy models, and submit cloud-based training through the SDK. Cloud-based functions require an AIE account and a generated call token.

#### Account registration
Please refer to the [AIE Document Center](https://engine-aiearth.aliyun.com/docs/page/guide?d=573e72) for account registration.


#### Token application
After logging in to the AIE account in the browser, visit the [token application page](https://engine-aiearth.aliyun.com/#/utility/auth-token) and generate a token according to the page prompts.

When calling aie.Authenticate() in the code, it will also prompt for a link to apply and wait for input. The effect is as follows:
```
Please paste the following address into the web browser, access the authorization page, and paste the personal token into the input box
         https://pre-engine-aiearth.aliyun.com/#/utility/auth-token
Personal token:
```

### Quick Start
Please refer to the [Quick Start Tutorial](quickstart.ipynb) for a quick start. We also provide more tutorials to facilitate your learning and use.

## Documentation

For different types of tasks, we provide specialized documentation for guidance.

### Data Processing

### Standard Algorithm Tasks
* [Land Cover Classification](aietorch/trainer/mmseg/configs/LandCover/README.md)
* [Land Cover Classification - Semi-Supervised Learning](aietorch/trainer/mmseg/configs/LandCover/README.md#configuration-4-semi-supervised-training-mkd)
* [Change Detection](aietorch/trainer/mmseg/configs/ChangeDet/README.md)
* [Building Change Detection](aietorch/trainer/mmseg/configs/BuildingChange/README.md)
* [Object Extraction](aietorch/trainer/mmseg/configs/TargetExtraction/README.md)

### Custom Algorithm Tasks
We also support users to customize their own algorithms based on the development paradigm provided in different Engines. For example, customizing model structure, loss functions, etc.
* [Developing Your Own Model](docs/source/custom_model.md)

### Cloud-based Training☁️

## Remote Sensing Model Zoo (Model Zoo)

| Task Type | Description | Model | Crop Size | mIoU | Config | Download |
| --- | --- | --- | --- | --- | --- | --- |
| ChangeDetection | General binary change detection | hrnet_w18 | 896x896 | N/A | [config](/aietorch/trainer/mmseg/configs/ChangeDet/hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25.py) | [model](xxx.pth) |
| ChangeDetection | Building change detection | hrnet_w18 | 896x896 | N/A | [config](/aietorch/trainer/mmseg/configs/BuildingChange/hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.py) | [model](xxx.pth) |
| LandCover | General land cover classification (11 classes) | hrnet_w48 | | N/A | [config](/aietorch/trainer/mmseg/configs/LandCover/fcn_hr48_1024x1024_16k_landcover.py) | [model](xxx.pth) |
| LandCover | Semi-supervised | hrnet_w48 | | N/A | [config](/aietorch/trainer/mmseg/configs/LandCover/semi.py) | N/A |
| TargetExtraction | Water body extraction | hrnet_w18 | | N/A | [config](/aietorch/trainer/mmseg/configs/TargetExtraction/fcn_hr18_1024x1024_40k4_bceious1w1.0.py) | [model](xxx.pth) |


## Open Source License

This project uses the [Apache 2.0 open source license](LICENSE). The project contains source code for some third-party dependencies, and some implementations are borrowed from other open source repositories. Please refer to the [NOTICE file](NOTICE) for the names of the repositories and the open source license descriptions.


## Contact Information

This project is maintained by the AI Earth team of Alibaba DAMO Academy. You can contact us through the following methods:

DingTalk group number: 

Email: 
