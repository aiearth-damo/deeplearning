## aiearth-deeplearning-0.0.2
* 仓库重命名为`deeplearning`，相关方法使用改为`from aiearth.deeplearning.xx import xx`。
* 添加云平台24中数据集作为内置的支持项。
* 增加4种预训练模型，可用于自定义微调。

## aiearth-deeplearning-0.0.3
* 移除mmseg库的相关源代码，改为pip依赖，减轻SDK，重构相关代码。
* 新增mmdet等依赖，加入建筑物矢量提取（检测类任务）PolyBuilding、多光谱预训练模型（分类任务）等面向遥感的多种任务类型。
