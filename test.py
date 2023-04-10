import numpy as np
from model import Generator,Discriminator
import torch.nn.functional as F
from matplotlib import pyplot as plt

import torch

# 主线程不建议写在 if外部。
if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    discriminator = Discriminator().to(device)
    discriminator.load_state_dict(torch.load('E:\desktop\dis_model_weights_30.pth'))
    discriminator.eval()

    generator = Generator().to(device)
    generator.load_state_dict(torch.load('E:\desktop\gen_model_weights_30.pth'))
    generator.eval()
    noise = torch.normal(mean=0, std=1, size=[10, 100]).to(device)

    gen = generator(noise)
    score = discriminator(gen)

    gen = (gen.cpu().detach().numpy() + 1) / 2
    # print(gen.shape)


    score = torch.flatten(score)
    gen_loss = F.mse_loss(torch.ones_like(score), score)
    print(torch.ones_like(score))
    print(score)
    print(gen_loss)

    plt.imshow(gen[0].squeeze(), cmap='gray')
    plt.show()
