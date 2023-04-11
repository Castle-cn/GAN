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


d = DiscriminatorLoss()
g = GeneratorLoss()

for i in range(2):
    data_1 = torch.randn(size=(32, 1),requires_grad=True)
    data_2 = torch.randn(size=(32, 1),requires_grad=True)

    loss_1 = d(data_1,data_2)
    loss_1.backward()
    loss_2 = g(data_2)
    loss_2.backward()



# loss_2 = g(data_2)
# loss_2.backward()