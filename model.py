import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch import nn


class BottleNeck(nn.Module):
    expension = 4  # 主要用在最后全连接神经网络计算输入维度

    def __init__(self, in_channels, out_channels, stride, downsample):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expension,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels * self.expension)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        print(x.shape)
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x) + identity
        x = self.relu(x)
        return x


class BuildingBlock(nn.Module):
    expension = 1  # 主要用在最后全连接神经网络计算输入维度

    def __init__(self, in_channels, out_channels, stride, downsample):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv_1(x)
        x = self.conv_2(x) + identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, block_nums, channels, classes_nums):
        super().__init__()

        self.layer_1 = self.make_layers(block, block_nums[0], 64, 64, 1)
        self.layer_2 = self.make_layers(block, block_nums[1], 64 * block.expension, 128, 2)
        self.layer_3 = self.make_layers(block, block_nums[2], 128 * block.expension, 256, 2)
        self.layer_4 = self.make_layers(block, block_nums[3], 256 * block.expension, 512, 2)

        self.stack = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *self.layer_1,
            *self.layer_2,
            *self.layer_3,
            *self.layer_4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(block.expension * 512, classes_nums),
            # nn.Softmax(dim=1)
        )

    def make_layers(self, block, block_nums, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expension:
            downsample = nn.Conv2d(in_channels, out_channels * block.expension, kernel_size=1,
                                   stride=stride, padding=0)

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_nums - 1):
            layers.append(block(out_channels * block.expension, out_channels, 1, None))
        return layers

    def forward(self, x):
        # print(x.shape)
        x = self.stack(x)
        return x


class LeNet(nn.Module):
    def __init__(self, in_channels, class_nums):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=120)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=class_nums)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, input):
        conv1_output = self.conv1(input)  # [28,28,1]-->[24,24,6]-->[12,12,6]
        conv2_output = self.conv2(conv1_output)  # [12,12,6]-->[8,8,16]-->[4,4,16]
        conv2_output = conv2_output.view(-1, 4 * 4 * 16)  # [n,4,4,16]-->[n,4*4*16],其中n代表个数
        fc1_output = self.fc1(conv2_output)  # [n,256]-->[n,120]
        fc2_output = self.fc2(fc1_output)  # [n,120]-->[n,84]
        fc3_output = self.fc3(fc2_output)  # [n,84]-->[n,10]
        out = self.sigmod(fc3_output)
        return out


class SimpleNet(nn.Module):
    def __init__(self, in_feature, num_class):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.reshape(x, [B, C * H * W])
        return self.stack(x)


class Generator(nn.Module):
    def __init__(self, img_size: list, batch_size):
        super().__init__()
        if len(img_size) > 2:
            raise ValueError("img_size should be (H,W) of the train image")
        self.H, self.W = img_size
        self.batch_size = batch_size
        self.generator = ResNet(BuildingBlock, [2, 2, 2, 2], 1, self.H * self.W)

    # 输入x是随机噪声，大小为224*224
    def forward(self, x):
        x = self.generator(x)  # 输出的就是图片打平后
        x = torch.round(F.normalize(x, dim=1) * 255)
        return x.reshape((self.batch_size, 1, self.H, self.W))  # [batch,1,H,W]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = LeNet(1, 1)
        # self.discriminator = SimpleNet(28 * 28, 1)

    def forward(self, x):
        x = self.discriminator(x)
        return x  # x is a scalar

# gen_model = Generator((28, 28))
