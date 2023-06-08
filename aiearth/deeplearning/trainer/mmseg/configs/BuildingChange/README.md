# 建筑物变化检测

## 任务描述

建筑物变化检测是一个遥感图像常见的任务，接收同源双图输入，给出图像中的变化区域，同时得到建筑物变化的类型，如新增建筑、建筑拆除、建筑改建等。

## 配置文件介绍及用法

[配置文件](hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.py)可以用于从头训练一个建筑物变化检测模型。我们已经提供了一个训练好的默认模型：`aie://BuildingChange/buildingchange_hrnet_w18_base_150k_new512_cosine_lr_batch_48_builingchange.pth`。配置默认有三种变化类型，分别为新增建筑、建筑拆除、建筑改建。
如果你有更进一步的需求，比如继续在你自己的数据集上进行模型微调，可以把我们提供的模型配置到load_from参数；如果你想新增类目，可以把我们的模型加载为预训练模型（model.pretrained），并修改配置文件中的`num_classes`设置所需要的变化类别数量。需要注意的是，`无变化`需要占用一个输出类别。
```python
model = dict(
    decode_head=dict(
        num_classes=4
    )
)
```
