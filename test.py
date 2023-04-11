import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Discriminator, Generator

if __name__ == '__main__':
    # d = Discriminator()
    g = Generator()

    # d.load_state_dict(torch.load(r'E:\desktop\dis_model_weights_81.pth'))
    g.load_state_dict(torch.load(r'E:\desktop\gen_model_weights_80.pth'))

    data = torch.randn(size=(10, 10))
    out = np.squeeze(g(data).detach().numpy())
    print(out.shape)
    out = (out + 1) / 2
    plt.imshow(out[1], cmap='gray')
    plt.show()