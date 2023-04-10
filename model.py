from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.generator = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 28 * 28),
            nn.Tanh(),
            nn.BatchNorm1d(28 * 28)
        )  # 输出来的在[-1,1]之间

    # 输入x是随机噪声
    def forward(self, x):
        x = self.generator(x)  # 输出的就是图片打平后
        return x.reshape((-1, 1, self.H, self.W))  # [batch,1,H,W]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.discriminator = LeNet(1, 1)
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )  # 输出的在[0,1]之间

    def forward(self, x):
        x = self.discriminator(x)
        return x  # x is a scalar

# gen_model = Generator((28, 28))
