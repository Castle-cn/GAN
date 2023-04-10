from torch import nn
import torch


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

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    # forward method
    def forward(self, x):
        x = self.stack(x)
        return x
