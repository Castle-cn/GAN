import os
import struct
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset
import numpy as np


class MnistDataset(Dataset):
    # 类的初始化,没什么好说的,固定格式,微调即可
    def __init__(self, data_root, transform=None):
        # self.train_img_dir = os.path.join(data_root, 'train-images.idx3-ubyte')
        self.test_img_dir = os.path.join(data_root, 't10k-images.idx3-ubyte')
        self.images = np.concatenate(self.run_pool(), axis=0)
        self.transform = transform

    # 获取数据集大小
    def __len__(self):
        return len(self.images)

    # 获取指定index的数据
    def __getitem__(self, idx):
        image = self.images[idx].reshape([28, 28])
        # transform固定写法, 基本不变
        if self.transform:
            image = self.transform(image)
        return image

    def read_image(self, image_path):
        with open(image_path, 'rb') as imgpath:
            _, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(num, rows * cols)
        return images

    def run_pool(self):
        cpu_worker_num = 4
        # process_args = [self.train_img_dir, self.test_img_dir]
        process_args = [self.test_img_dir]
        with Pool(cpu_worker_num) as p:
            outputs = p.map(self.read_image, process_args)
        return outputs


class NoiseDataset(Dataset):
    def __init__(self, mean, std, nums, img_size: list):
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.nums = nums
        self.images = torch.cat(self.run_pool(), dim=0)
        self.images = self.images[:, None, ...]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image

    def gen_noise(self, args):
        args['size'].insert(0, args['num'])
        return torch.normal(mean=self.mean, std=self.std, size=args['size'])

    def run_pool(self):
        cpu_worker_num = 4
        n = self.nums // 10000
        process_args = [{'num': 10000, 'size': self.img_size}] * n
        if self.nums % 10000 != 0:
            process_args.append({'num': self.nums % 10000, 'size': self.img_size})

        with Pool(cpu_worker_num) as p:
            outputs = p.map(self.gen_noise, process_args)
        return outputs


# class GenImage(Dataset):
#     # img_size 是生成的图片的大小
#     def __init__(self, gen_images, img_size: list):
#         mean = torch.mean(gen_images)
#         std = torch.std(gen_images)
#         gen_images = (gen_images - mean) / std
#         self.images = gen_images.reshape((len(gen_images), 1, img_size[0], img_size[1]))
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, index):
#         image = self.images[index]
#         return image
