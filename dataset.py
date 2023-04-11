import os
import struct
import torch
import torchvision.transforms
from torch.utils.data import Dataset
import numpy as np


def read_image(image_path):
    with open(image_path, 'rb') as imgpath:
        _, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(num, rows, cols)
    return images


class MnistDataset(Dataset):
    # 类的初始化,没什么好说的,固定格式,微调即可
    def __init__(self, data_root, transform=None):
        # 改这里记得把train里面noise生成的数目改了
        self.train_img_dir = os.path.join(data_root, 'train-images.idx3-ubyte')
        self.test_img_dir = os.path.join(data_root, 't10k-images.idx3-ubyte')
        self.images = np.concatenate([read_image(self.test_img_dir), read_image(self.train_img_dir)])
        # self.images = np.concatenate([read_image(self.test_img_dir)])
        self.transform = transform

    # 获取数据集大小
    def __len__(self):
        return len(self.images)

    # 获取指定index的数据
    def __getitem__(self, idx):
        image = self.images[idx]
        # transform固定写法, 基本不变
        if self.transform:
            image = self.transform(image)
        return image
