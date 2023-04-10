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
    a = np.asarray([[7.5476e-05],
                    [4.9661e-05],
                    [8.0105e-05],
                    [1.1499e-04],
                    [7.5657e-05],
                    [7.2256e-05],
                    [7.2581e-05],
                    [3.9100e-05],
                    [7.3301e-05],
                    [2.7205e-05],
                    [4.4697e-05],
                    [3.1211e-05],
                    [3.2596e-05],
                    [4.8518e-05],
                    [4.2086e-05],
                    [3.8372e-05],
                    [4.4195e-05],
                    [5.8393e-05],
                    [8.7155e-05],
                    [8.2982e-05],
                    [3.2820e-05],
                    [6.0196e-05],
                    [6.7776e-05],
                    [3.6987e-05],
                    [3.8355e-05],
                    [3.0987e-05],
                    [4.6070e-05],
                    [7.9648e-05],
                    [6.8070e-05],
                    [9.0252e-05],
                    [2.1299e-05],
                    [6.5475e-05]])
    a = np.squeeze(np.reshape(a, [1, -1]))
    b = np.zeros_like(a)
    print(cross_entropy(a, b))

    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    print(F.cross_entropy(a, b))
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
