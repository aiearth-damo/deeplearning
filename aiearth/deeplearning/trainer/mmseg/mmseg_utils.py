import os
from mmcv.utils import Config
from .vars import CONFIG_BASE_DIR, SCHEDULE_CONFIG_PATH, RUNTIME_CONFIG_PATH


class AIEToMMsegCoverter(object):
    def to_mmseg_dataset_cfg(self, mmseg_dataset_dict:dict, mmseg_dataset_cfg_template_path:str):
        ''' mmseg_dataset_dict:内容如下
              dataset: 数据集加载类目名称
              img_dir: 影像所在目录
              img_suffix: 影像文件路径后缀
              ann_dir: 标注所在目录
              seg_map_suffix: 标注文件路径后缀
              split: 影像与标注文件映射文件路径，内容格式，
                     单图：
                     image_path ann_path
                     双图：
                     image_1_path image_2_path ann_path
              classes: 类目列表
              '''
        datasets = Config.fromfile(mmseg_dataset_cfg_template_path)
        #custom_datasets = datasets.__getattribute__('_cfg_dict').to_dict()
        tmp_datasets = dict(
            data=dict(
                train = dict(data_root=mmseg_dataset_dict["img_dir"], split=mmseg_dataset_dict["split"]),
                val = dict(data_root=mmseg_dataset_dict["img_dir"], split=mmseg_dataset_dict["split"]),
                test = dict(data_root=mmseg_dataset_dict["img_dir"], split=mmseg_dataset_dict["split"]),
            )
        )
        datasets.merge_from_dict(tmp_datasets)
        return datasets


def mmseg_base_config():
    base_config = Config()
    runtime_config = Config.fromfile(RUNTIME_CONFIG_PATH)
    base_config.merge_from_dict(runtime_config.__getattribute__('_cfg_dict').to_dict())
    schedule_config = Config.fromfile(SCHEDULE_CONFIG_PATH)
    base_config.merge_from_dict(schedule_config.__getattribute__('_cfg_dict').to_dict())
    return base_config

def to_mmseg_config(config_name):
    base_config = mmseg_base_config()
    model_config_path = os.path.join(CONFIG_BASE_DIR, config_name + ".py")
    model_config = Config.fromfile(model_config_path)
    model_config.merge_from_dict(base_config.__getattribute__('_cfg_dict').to_dict())
    return model_config


