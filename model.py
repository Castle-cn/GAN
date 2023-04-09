import torch
from torch import nn
from torchvision import models


class Generator(nn.Module):
    def __init__(self, img_size: tuple):
        super().__init__()
        if len(img_size) > 2:
            raise ValueError("img_size should be (H,W) of the train image")
        self.H, self.W = img_size
        self.generator = models.resnet50(pretrained=False,
                                         num_classes=self.H * self.W)

    # 输入x是随机噪声，大小为224*224
    def forward(self, x):
        x = self.generator(x)  # 输出的就是图片打平后
        return x.reshape((self.H, self.W))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = models.resnet50(pretrained=False,
                                         num_classes=1)

    def forward(self,x):
        x = self.discriminator(x)
        return x