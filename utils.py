from torch import nn
import torch
import torch.nn.functional as F


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_img_score, gen_img_score):
        loss = torch.mean(real_img_score - gen_img_score)
        return -loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_img_score):
        gen_loss = torch.mean(gen_img_score)
        return -gen_loss
