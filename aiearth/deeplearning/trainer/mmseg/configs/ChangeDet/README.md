# 变化检测

## 任务描述

变化检测是一个遥感图像常见的任务，需要同源双图输入，给出图像中的内容变化，例如通用变化检测、建筑物变化检测、水体变化检测、耕地变化检测等。

## 配置文件介绍及用法

我们提供了一整套模型配置和默认参数，能够有效训练出性能优秀的模型。你可以利用自建的变化检测数据集，在该框架下训练所需的变化检测任务模型。
### 配置1
[effi-b0_base_50k_new256_cosine_lr_batch_128_adamw](effi-b0_base_50k_new256_cosine_lr_batch_128_adamw.py)：
使用该配置，可以训练出一个backbone为efficient-b0的轻量变化检测模型，该模型兼顾性能和效果。如果你有自己的需要，可以参考此配置文件完成一次从头训练。

### 配置2
[hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25](hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25.py)：
使用该配置，可以训练出一个sota的hrnet18模型。我们已经提供的sota[模型](xxx.pth)就是基于该配置和相关数据训练而成。或者如果你有自己的需要，可以参考此配置文件完成一次从头训练。

### 配置3
[hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune](hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune.py)：
我们也提供了一套用于微调的配置。使用该配置，配合前序模型，你可以用在自己的数据上进行模型微调，以取得在你的数据上更具表现力的效果。

### 配置4
[hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_semi](hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_semi.py):
使用该配置可以引入额外的无标签数据对模型进行半监督训练，提升模型的表现能力。
使用方法：
#### 步骤 1
使用[配置2](hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25.py)或者[配置3](hrnet_w18_base_150k_new512_cosine_lr_batch_48_v25_finetune.py)得到一个sota表现的模型以后，通过[推理](/quickstart.ipynb#Test)功能保存无标签数据的伪标签。
