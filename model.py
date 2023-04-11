from torch import nn
import numpy as np

img_size = [1, 28, 28]


class Generator(nn.Module):
    # 输入的是随机噪声
    def __init__(self, in_dims=10):
        super(Generator, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(in_dims, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, np.prod(img_size).item()),
            nn.Tanh()  # 输出在-1到1之间
        )

    def forward(self, x):
        x = self.stack(x)
        x = x.reshape([x.shape[0], *img_size])
        return x


class Discriminator(nn.Module):
    # 输入是一张图片
    def __init__(self):
        super(Discriminator, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(np.prod(img_size).item(), 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入是一张图片
        x = x.reshape([x.shape[0], -1])
        x = self.stack(x)
        return x
