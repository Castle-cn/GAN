from torch import nn
import torch


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_img_score, gen_img_score):
        v_real = torch.log(real_img_score).mean()
        v_gen = torch.log(1 - gen_img_score).mean()
        return -(v_real + v_gen)


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_img_score):
        v_gen = torch.log(gen_img_score).mean()
        return -v_gen
