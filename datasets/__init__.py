# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .market1501 import Market1501
from .mars import Mars
from .dataset_loader import ImageDataset
from .dataset_loader import VideoDataset
from .samplers import  RandomIdentitySampler

__factory = {
    'market1501': Market1501,
    'mars': Mars,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
