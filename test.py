import numpy as np
from model import Generator, Discriminator
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch

if __name__ == '__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    discriminator = Discriminator(28 * 28).to(device)
    discriminator.load_state_dict(torch.load('E:\desktop\dis_model_weights_81.pth'))
    discriminator.eval()

    generator = Generator(100, 28 * 28).to(device)
    generator.load_state_dict(torch.load('E:\desktop\gen_model_weights_81.pth'))
    generator.eval()
    noise = torch.normal(mean=0, std=1, size=[10, 100]).to(device)

    gen = generator(noise)
    score = discriminator(gen)

    gen = (gen.cpu().detach().numpy() + 1) / 2
    # print(gen.shape)

    score = torch.flatten(score)
    gen_loss = F.cross_entropy(score,torch.ones_like(score))

    print(score)
    print(gen_loss)

    plt.imshow(gen[0].reshape((28, 28)), cmap='gray')
    plt.show()
