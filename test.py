import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from model import Discriminator, Generator

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

if __name__ == '__main__':
    # d = Discriminator()
    g = Generator()

    # d.load_state_dict(torch.load(r'E:\desktop\dis_model_weights_81.pth'))
    g.load_state_dict(torch.load(r'model/gen_model_weights_80.pth'))

    data = torch.randn(size=(10, 10))
    out = g(data)
    # print(out.shape)
    out = (out + 1) / 2
    grid = make_grid(out)
    show(grid)
    plt.tight_layout()
    plt.show()