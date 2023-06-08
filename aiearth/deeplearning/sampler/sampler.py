import abc
import random
from copy import copy
from typing import Optional
from aiearth.deeplearning.datasets.datasets import Dataset, NonGeoDataset

def split_list_by_percent(lst, percent):
    list_len = len(lst)
    split_index = int(list_len * percent)
    return lst[0:split_index], lst[split_index:list_len]


class DatasetSampler(metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def img_ann_mapping_list_handler(cls, dataset: Dataset) -> list:
        pass

    @classmethod
    def sample_by_count(cls, dataset: Dataset, count: int, start_idx=0) -> Dataset:
        new_dataset = copy(dataset)
        tmp_mapping_lst = cls.img_ann_mapping_list_handler(dataset)  
        new_img_ann_mapping_list =  tmp_mapping_lst[0: count]
        new_dataset.img_ann_mapping_list = new_img_ann_mapping_list
        return new_dataset
    
    @classmethod
    def split_by_count(cls, dataset: Dataset, count: int) -> list:
        tmp_mapping_lst = cls.img_ann_mapping_list_handler(dataset)
        list_1 = tmp_mapping_lst[0: count]
        list_2 = tmp_mapping_lst[count, len(tmp_mapping_lst)]
        dataset_1 = copy(dataset)
        dataset_2 = copy(dataset)
        dataset_1.img_ann_mapping_list = list_1
        dataset_2.img_ann_mapping_list = list_2
        return [dataset_1, dataset_2]

    @classmethod 
    def split_by_percent(cls, dataset: Dataset, percent:float) -> list:
        new_img_ann_mapping_list = cls.img_ann_mapping_list_handler(dataset)
        list_1, list_2 = split_list_by_percent(new_img_ann_mapping_list, percent)
        dataset_1 = copy(dataset)
        dataset_2 = copy(dataset)
        dataset_1.img_ann_mapping_list = list_1
        dataset_2.img_ann_mapping_list = list_2
        return [dataset_1, dataset_2]

    @classmethod
    def sample_by_percent(cls, dataset: Dataset, percent:float) -> Dataset:
        new_img_ann_mapping_list = cls.img_ann_mapping_list_handler(dataset)
        list_1, _ = split_list_by_percent(new_img_ann_mapping_list, percent)
        new_dataset = copy(dataset)
        new_dataset.img_ann_mapping_list = list_1
        return new_dataset



class RandomNonGeoDatasetSampler(DatasetSampler):

    @classmethod
    def img_ann_mapping_list_handler(cls, dataset: Dataset) -> list:
        new_lst = copy(dataset.img_ann_mapping_list)
        random.shuffle(new_lst)
        return new_lst

class SequentialNonGeoDatasetSampler(DatasetSampler):

    @classmethod
    def img_ann_mapping_list_handler(cls, dataset: Dataset) -> list:
        new_lst = copy(dataset.img_ann_mapping_list)
        return new_lst


        

        