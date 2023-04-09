import os
import struct

import torch
from torch.utils.data import Dataset
import numpy as np


def read_image(image_path):
    with open(image_path, 'rb') as imgpath:
        _, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(num, rows * cols)
    return rows, cols, torch.from_numpy(images)


class MnistDataset(Dataset):
    # 类的初始化,没什么好说的,固定格式,微调即可
    def __init__(self, data_root, transform=None):
        train_img_dir = os.path.join(data_root, 'train-images.idx3-ubyte')
        test_img_dir = os.path.join(data_root, 't10k-images.idx3-ubyte')

        self.rows, self.cols, train_images = read_image(train_img_dir)
        _, _, test_images = read_image(test_img_dir)

        self.images = torch.cat((train_images, test_images), 0)
        self.transform = transform

    # 获取数据集大小
    def __len__(self):
        return len(self.images)

    # 获取指定index的数据
    def __getitem__(self, idx):
        image = self.images[idx].reshape([self.rows, self.cols])
        # transform固定写法, 基本不变
        if self.transform:
            image = self.transform(image)
        return image


class NoiseDataset(Dataset):
    def __init__(self, mean, std, nums, img_size: list):
        self.images = torch.normal(mean, std, size=(nums, *img_size))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image
