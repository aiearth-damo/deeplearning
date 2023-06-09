# AIE云上样本集使用


## 前期准备
### 概念介绍
#### 1）样本集ID
在AIE云平台模型训练->样本管理系统中，所管理的样本集均有唯一的ID值，可通过ID精确匹配目标样本集。样本集ID可通过页面查看样本集详情时，URL路径的最后一个路径的数字既为样本id


```
URL: https://engine-aiearth.aliyun.com/#/portal/model/sample/detail/600054?pageNo=1

样本集ID: 600054
```

#### 2) 样本集名称
在AIE云平台模型训练->样本管理系统中，所管理的样本集均有名称，既前端页面样本名称列所展示的字段。该名称可重复，可在SDK中通过名称进行搜索，返回值是一个列表，需要用户自行判断。

## quick start

### 配置AIE平台Token


以下步骤全局设置一次就可以


```
import aie
# Authenticate无参数会进入交互式shell输入token
aie.Authenticate()


# Authenticate也可以直接配置token
# token = 'xxxxxxxxxxxx'
# aie.Authenticate(token)
```


### 地物分类样本集

```
# 地物分类样本集
from aiearth.deeplearning.cloud.datasets import LandcoverDataset, PublicDatasetMeta


# 使用dataset id获取云上样本集
dataset = LandcoverDataset(PublicDatasetMeta.GID_15_TRAIN["dataset_id"], data_root="./landcover")

print(dataset.classes)
# output
# ['背景', 'industrial_land', 'garden_land', 'urban_residential', 'arbor_forest', 'rural_residential', 'shrub_land', 'traffic_land', 'natural_meadow', 'paddy_field', 'artificial_meadow', 'irrigated_land', 'river', 'dry_cropland', 'lake', 'pond']

# 提取某些类目，其他类目作为背景
dataset.set_classes_filter(['industrial_land'])
print(dataset.classes)
# output
# ['background', 'industrial_land']

# 在样本集初始化时直接提取类目
dataset = LandcoverDataset(PublicDatasetMeta.GID_15_TRAIN["dataset_id"], data_root="./landcover", classes_filter=['industrial_land'])
```


### 变化检测样本集(二分类)



```
from aiearth.deeplearning.cloud.datasets import BinaryChangeDetDataset, PublicDatasetMeta

# 使用dataset id获取云上样本集
dataset = BinaryChangeDetDataset(PublicDatasetMeta.SEMANTIC_CHANGE_DETECTION_SECOND["dataset_id"], data_root="./changedet")
```



### 目标提取（地物识别）


```
from aiearth.deeplearning.cloud.datasets import TargetExtractionDataset, PublicDatasetMeta

# 使用dataset id获取云上样本集
dataset = TargetExtractionDataset(PublicDatasetMeta.BUILDING_AERIAL_IMAGERY_TRAIN["dataset_id"], data_root="./target_extraction")
```
