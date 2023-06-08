# AIE SDK

English | [简体中文](README.md)

## Introduction

AIE SDK is a deep learning training framework for remote sensing image processing. It provides local and cloud-based capabilities, rich data set acquisition, ease of use, and scalability.

## Installation

```
# for devel
pip install -e .

# build package and install by package
bash scripts/build_pkg.sh
pip install dist/aiearth-deeplearning.tar.gz
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
         https://engine-aiearth.aliyun.com/#/utility/auth-token
Personal token:
```

### Quick Start
Please refer to the [Quick Start Tutorial](quickstart.ipynb) for a quick start. We also provide more tutorials to facilitate your learning and use.

## Documentation

For different types of tasks, we provide specialized documentation for guidance.

### Dataset Processing
* [cloud-based dataset](docs/dataset/cloud.md)
* [custom dataset](docs/dataset/custom.md)

### Create training job
* [training job](docs/train/train.md)

### Standard Algorithm Tasks
* [Land Cover Classification](aiearth/deeplearning/trainer/mmseg/configs/LandCover/README.md)
* [Land Cover Classification - Semi-Supervised Learning](aiearth/deeplearning/trainer/mmseg/configs/LandCover/README.md#configuration-4-semi-supervised-training-mkd)
* [Change Detection](aiearth/deeplearning/trainer/mmseg/configs/ChangeDet/README.md)
* [Building Change Detection](aiearth/deeplearning/trainer/mmseg/configs/BuildingChange/README.md)
* [Object Extraction](aiearth/deeplearning/trainer/mmseg/configs/TargetExtraction/README.md)

### Custom Algorithm Tasks
We also support users to customize their own algorithms based on the development paradigm provided in different Engines. For example, customizing model structure, loss functions, etc.
* [Developing Your Own Model](docs/model/custom_model.md)

## Remote Sensing Model Zoo (Model Zoo)

| Task Type        | Description                                    | Model     | Crop Size | mIoU | Config                                                                                                               |
| ---------------- | ---------------------------------------------- | --------- | --------- | ---- | -------------------------------------------------------------------------------------------------------------------- |
| ChangeDetection  | General binary change detection                | hrnet_w18 | 896x896   | N/A  | [config](aiearth/deeplearning/trainer/mmseg/configs/ChangeDet/hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25.py)                |
| ChangeDetection  | Building change detection                      | hrnet_w18 | 896x896   | N/A  | [config](aiearth/deeplearning/trainer/mmseg/configs/BuildingChange/hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.py) |
| LandCover        | General land cover classification (11 classes) | hrnet_w48 |           | N/A  | [config](aiearth/deeplearning/trainer/mmseg/configs/LandCover/fcn_hr48_1024x1024_16k_landcover.py)                                 |
| LandCover        | Semi-supervised                                | hrnet_w48 |           | N/A  | [config](aiearth/deeplearning/trainer/mmseg/configs/LandCover/semi.py)                                                             |
| TargetExtraction | Water body extraction                          | hrnet_w18 |           | N/A  | [config](aiearth/deeplearning/trainer/mmseg/configs/TargetExtraction/fcn_hr18_1024x1024_40k4_bceious1w1.0.py)                      |


## Open Source License

This project uses the [Apache 2.0 open source license](LICENSE). 

## Contact Information

This project is maintained by the AI Earth team of Alibaba DAMO Academy. You can contact us through the following methods:

| DingTalk    | WeChat  |Email  
| :----------- | :-----------: |:-----------: |
| DingTalk group number: 32152986 | WeChat public account: AI Earth数知地球 |aiearth@service.aliyun.com
| ![钉钉群号](https://img.alicdn.com/imgextra/i2/O1CN01XW3sCk1JlBoQ5tKAd_!!6000000001068-2-tps-159-160.png "钉钉群号") | ![钉钉群号](https://img.alicdn.com/imgextra/i2/O1CN0109JceF1W63CuznFtA_!!6000000002738-2-tps-160-160.png "钉钉群号") |
