# 用户自定义本地数据集

## NonGeoCustomDataset(无地理信息数据集)
该数据集的样本类型为无地理信息的普通图像，图像支持jpg和png，标注格式根据不同的样本集类型有差异。

### 变化检测数据集(二分类)
标注文件规范：

- 文件格式：单通道png
- 数据形式：类目由该类目在classes中索引值表示。如classes为['background', 'change']，则background在影像中用像素值0表示，change在影像中用像素值1表示。
```python
from aiearth.deeplearning.datasets import ChangeDetNonGeoCustomDataset

dataset = ChangeDetNonGeoCustomDataset(
    classes=['background', 'change'],
    img_shape=(512, 512, 3),
    ann_shape=(512, 512, 1),
    img_dir="datasets/images1",
    img2_dir="datasets/images2", 
    img_suffix=".jpg",
    ann_dir="datasets/annotations", 
    ann_suffix=".png",
)
```
参数说明

| 参数名称 | 参数类型 | 参数说明 | 默认值 |
| --- | --- | --- | --- |
| classes | list[str] | 类目列表 | 无，必填 |
| img_shape | tuple or list | 影像的数据信息三元数组，分别是（长，宽，通道数），对应影像转numpy array后 shape的三元组,目前仅支持rgb 3通道的影像 | 无，必填 |
| ann_shape | tuple or list | 标注数据三元数组，分别是（长，宽，通道数）,对应影像转numpy array后 shape的三元组, 目前仅支持单通道的灰度图 | 无，必填 |
| img_dir | str | 变化前期图像的目录 | None |
| img2_dir | str | 变化后期图像的目录 | None |
| ann_dir | str | 标注数据目录 | None |
| img_suffix | str | 文件扩展名 | .jpg  |
| ann_suffix | str | 标注文件格式 | .png |
| img_ann_mapping_list | list[dict] | 影像，标注信息映射list，格式如下[{"img": before1_jpg,"img2": after1_jpg,"ann": ann1_path,}，{"img": before2_jpg,"img2": after2_jpg,"ann": ann2_path,}，...]如果该参数为None，则根据img_dir, img2_dir, ann_dir进行文件扫描，同名文件会自动按照上述格式生成img_ann_mapping_list，如果文件名称有差异，则需要自行构建该参数，指定映射关系。 | None |

### 地物分类数据集
标注文件规范：

- 文件格式：单通道png
- 数据形式：类目由该类目在classes中索引值表示。如classes为['industrial_land', 'garden_land', 'arbor_forest']，则industrial_land在影像中用像素值0表示，garden_land在影像中用像素值1表示，以此类推
- 特殊说明：地物分类样本需全像素标注，未标注区域自动转换成像素0，像素值255为待定区域，训练时忽略该区域。
```python
from aiearth.deeplearning.datasets import LandcoverNonGeoCustomDataset


classes = ['industrial_land',
 'garden_land',
 'urban_residential',
 'arbor_forest',
 'rural_residential',
 'shrub_land',
 'traffic_land',
 'natural_meadow',
 'paddy_field',
 'artificial_meadow',
 'irrigated_land',
 'river',
 'dry_cropland',
 'lake',
 'pond'
]

dataset = LandcoverNonGeoCustomDataset(
    classes=classes,
    img_shape=(512, 512, 3),
    ann_shape=(512, 512, 1),
    img_dir="datasets/images1",
    img_suffix=".jpg",
    ann_dir="datasets/annotations", 
    ann_suffix=".png",
)
```
| 参数名称 | 参数类型 | 参数说明 | 默认值 |
| --- | --- | --- | --- |
| classes | list[str] | 类目列表 | 无，必填 |
| img_shape | tuple or list | 影像的数据信息三元数组，分别是（长，宽，通道数）,对应影像转numpy array后 shape的三元组,目前仅支持rgb 3通道的影像 | 无，必填 |
| ann_shape | tuple or list | 标注数据三元数组，分别是（长，宽，通道数），对应影像转numpy array后 shape的三元组，目前仅支持单通道的灰度图 | 无，必填 |
| img_dir | str | 图像的目录 | None |
| ann_dir | str | 标注数据目录 | None |
| img_suffix | str | 文件扩展名 | .jpg  |
| ann_suffix | str | 标注文件格式 | .png |
| img_ann_mapping_list | list[dict] | 影像，标注信息映射list，格式如下[{"img": img_path_1,"ann": ann1_path,}，{"img": img_path_2,"ann": ann2_path,}，...] 如果该参数为None，则根据img_dir, ann_dir进行文件扫描，同名文件会自动按照上述格式生成img_ann_mapping_list，如果文件名称有差异，则需要自行构建该参数，指定映射关系。 | None |

### 目标提取（地物识别）数据集
标注文件规范：

- 文件格式：单通道png
- 数据形式：类目由该类目在classes中索引值表示。如classes为['background', 'target']，则background在影像中用像素值0表示，target在影像中用像素值1表示。
```python
from aiearth.deeplearning.datasets import TargetExtractionNonGeoCustomDataset


classes = ['background', 'target']

dataset = TargetExtractionNonGeoCustomDataset(
    classes=classes,
    img_shape=(512, 512, 3),
    ann_shape=(512, 512, 1),
    img_dir="datasets/images1",
    img_suffix=".jpg",
    ann_dir="datasets/annotations", 
    ann_suffix=".png",
)
```
| 参数名称 | 参数类型 | 参数说明 | 默认值 |
| --- | --- | --- | --- |
| classes | list[str] | 类目列表 | 无，必填 |
| img_shape | tuple or list | 影像的数据信息三元数组，分别是（长，宽，通道数），对应影像转numpy array后 shape的三元组，目前仅支持rgb 3通道的影像 | 无，必填 |
| ann_shape | tuple or list | 标注数据三元数组，分别是（长，宽，通道数），对应影像转numpy array后 shape的三元组，目前仅支持单通道的灰度图 | 无，必填 |
| img_dir | str | 图像的目录 | None |
| ann_dir | str | 标注数据目录 | None |
| img_suffix | str | 文件扩展名 | .jpg  |
| ann_suffix | str | 标注文件格式 | .png |
| img_ann_mapping_list | list[dict] | 影像，标注信息映射list，格式如下[{"img": img_path_1,"ann": ann1_path,}，{"img": img_path_2,"ann": ann2_path,}，...] 如果该参数为None，则根据img_dir, ann_dir进行文件扫描，同名文件会自动按照上述格式生成img_ann_mapping_list，如果文件名称有差异，则需要自行构建该参数，指定映射关系。 | None |

