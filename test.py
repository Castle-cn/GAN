import numpy as np
from model import Generator, Discriminator
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch


def cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)  # 防止取 log 时出现零值或负数
    ce = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return ce


if __name__ == '__main__':
    a = np.asarray([[7.2408e-05],
                    [6.4463e-05],
                    [5.4610e-05],
                    [2.7434e-05],
                    [6.6149e-05],
                    [4.8179e-05],
                    [3.9501e-05],
                    [6.2784e-05],
                    [8.6827e-05],
                    [5.0681e-05],
                    [3.7593e-05],
                    [2.9852e-05],
                    [7.2825e-05],
                    [7.9838e-05],
                    [5.8750e-05],
                    [3.5412e-05],
                    [3.9796e-05],
                    [6.4460e-05],
                    [2.4372e-05],
                    [4.3131e-05],
                    [6.7271e-05],
                    [3.8452e-05],
                    [4.6007e-05],
                    [6.5632e-05],
                    [7.0021e-05],
                    [1.5808e-05],
                    [4.0992e-05],
                    [7.8119e-05],
                    [3.2249e-05],
                    [8.5959e-05],
                    [4.9668e-05],
                    [8.7942e-05]])
    a = np.squeeze(np.reshape(a, [1,-1]))
    b = np.ones_like(a)
    print(cross_entropy(a,b))

    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    print(F.cross_entropy(a,b))
    # device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda'
    #
    # discriminator = Discriminator().to(device)
    # discriminator.load_state_dict(torch.load('E:\desktop\dis_model_weights_30.pth'))
    # discriminator.eval()
    #
    # generator = Generator().to(device)
    # generator.load_state_dict(torch.load('E:\desktop\gen_model_weights_30.pth'))
    # generator.eval()
    # noise = torch.normal(mean=0, std=1, size=[10, 100]).to(device)
    #
    # gen = generator(noise)
    # score = discriminator(gen)
    #
    # gen = (gen.cpu().detach().numpy() + 1) / 2
    # # print(gen.shape)
    #
    #
    # score = torch.flatten(score)
    # gen_loss = F.mse_loss(torch.ones_like(score), score)
    # print(torch.ones_like(score))
    # print(score)
    # print(gen_loss)
    #
    # plt.imshow(gen[0].squeeze(), cmap='gray')
    # plt.show()
