from torch import nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(g_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, g_output_dim),
            nn.Tanh(),
            nn.BatchNorm1d(g_output_dim)
        )

    # forward method
    def forward(self, x):
        x = self.stack(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.flat = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.sigmod = nn.Sigmoid()

        self.linear1 = nn.Linear(d_input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, 1)

        # self.stack = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(d_input_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )

    # forward method
    def forward(self, x):
        # x = self.stack(x)

        x = self.flat(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.linear4(x)
        x = self.sigmod(x)

        return x


# gen_model = Generator((28, 28))
model = Discriminator(28 * 28)
data = torch.randn((31, 28 * 28))
out = model(data)
