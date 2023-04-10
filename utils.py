from torch import nn
import torch
import torch.nn.functional as F

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_img_score, gen_img_score):
        real_loss = F.cross_entropy(torch.ones_like(real_img_score),real_img_score)
        gen_loss = F.cross_entropy(torch.zeros_like(gen_img_score),gen_img_score)
        return real_loss + gen_loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_img_score):
        gen_loss = F.cross_entropy(torch.ones_like(gen_img_score),gen_img_score)
        return gen_loss
