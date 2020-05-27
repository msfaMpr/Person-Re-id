# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class VideoDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, seq_len, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num_imgs = len(img_paths)

        # Evenly samples seq_len images from a tracklet
        if num_imgs >= self.seq_len:
            num_imgs -= num_imgs % self.seq_len
            indices = np.arange(0, num_imgs, num_imgs / self.seq_len)
        else:
            # if num_imgs is smaller than seq_len, simply replicate the last image
            # until the seq_len requirement is satisfied
            indices = np.arange(0, num_imgs)
            num_pads = self.seq_len - num_imgs
            indices = np.concatenate(
                [
                    indices,
                    np.ones(num_pads).astype(np.int32) * (num_imgs-1)
                ]
            )
        assert len(indices) == self.seq_len

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0) # img must be torch.Tensor
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid
