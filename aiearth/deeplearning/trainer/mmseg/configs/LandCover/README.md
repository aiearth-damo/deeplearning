# 地物分类

## 任务描述
用于多目标分割任务，例如土地使用分析，建筑物水体绿地等多目标分割等。

## 配置文件介绍
我们提供了一整套模型配置和默认参数，能够有效训练出性能优秀的模型。

### 配置1
[fcn_hr18_1024x1024_16k_landcover](fcn_hr18_1024x1024_16k_landcover.py)：
使用该配置，可以训练出一个效果很好的hrnet18模型。如果你有自己的需要，可以参考此配置文件完成一次从头训练。

### 配置2
[fcn_hr48_1024x1024_16k_landcover](fcn_hr48_1024x1024_16k_landcover.py)：
Backbone为hrnet48的模型。使用该配置，可以训练出一个效果很好的hrnet48模型。如果你有自己的需要，可以参考此配置文件完成一次从头训练。
另外，我们提供了基于该配置训练出来的的sota模型：`aie://LandCover/landcover_v1.6.pth`。用户可以通过该模型出发，完成在自己数据上的微调。

### 配置3与配置4
[fcn_hr18_1024x1024_16k_landcover_semi](fcn_hr18_1024x1024_16k_landcover_semi.py)
使用该配置可以引入额外的无标签数据对模型进行半监督训练，提升模型的表现能力。

[MKD](mkd_semi_sup.py)

近期半监督语义分割方法中一直在广泛研究一致性正则化。受益于图像、特征和网络扰动，取得了显著的性能。为了充分利用这些扰动，本文提出了一种新的一致性正则化框架，称为互相知识蒸馏（MKD）。我们创新地引入了基于一致性正则化方法的两个辅助均值教师模型。更具体地说，我们使用一位均值老师生成的伪标签来监督另一个学生网络，以实现两个分支之间的互相知识蒸馏。除了使用图像级强度和弱度增强之外，我们还考虑隐式语义分布采用特征增强来为学生添加更多扰动。所提出的框架显着增加了训练样本的多样性。在公共基准测试中的大量实验表明，我们的框架在不同的半监督设置下优于以前的最新技术（SOTA）方法。用于使用更多的无标签数据提升地物分类的性能。

#### 使用方法：
准备好有标签数据和无标签数据，直接使用[配置3](fcn_hr18_1024x1024_16k_landcover_semi.py)进行训练，即可使用我们提出的MKD方法。

#### 参见[MKD Offical Github](https://github.com/jianlong-yuan/semi-mmseg)

#### Citation

```bibtext
@article{yuan2022semi,
  title={Semi-supervised Semantic Segmentation with Mutual Knowledge Distillation},
  author={Yuan, Jianlong and Ge, Jinchao and Wang, Zhibin and Shen,Chunhua and Liu, Yifan},
  journal={arXiv preprint arXiv:2208.11499},
  year={2022}
}
```
