import torch
from torchvision import transforms
from model import Generator, Discriminator
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from torchvision.utils import save_image
from dataset import MnistDataset


class MyLoader:
    def __init__(self, real_data_root, batch_size):
        self.real_data_root = real_data_root
        self.batch_size = batch_size
        self.real_loader = self.get_dataloader()

    def get_dataloader(self):
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        real_data = MnistDataset(data_root=self.real_data_root, transform=transform)
        real_loader = DataLoader(real_data,
                                 batch_size=self.batch_size,
                                 drop_last=True)
        print('real data has been loaded over!!')

        return real_loader


class Model:
    def __init__(self, loader: MyLoader, device):
        self.loader = loader
        self.device = device
        self.lr = 1e-3
        self.noise_dims = 10
        self.loss_fn = nn.BCELoss()

        self.g_model = Generator(self.noise_dims).to(device)
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=self.lr)

        self.d_model = Discriminator().to(device)
        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=self.lr)

    def train(self):
        num_batches = len(self.loader.real_loader)
        with tqdm(total=num_batches) as pbar:
            for _, real_imgs in enumerate(self.loader.real_loader):
                real_imgs = real_imgs.to(self.device)
                noise = torch.randn(size=(self.loader.batch_size, self.noise_dims)).to(self.device)
                target_ones = torch.ones(self.loader.batch_size, 1).to(self.device)
                target_zeros = torch.zeros(self.loader.batch_size, 1).to(self.device)

                fake_imgs = self.g_model(noise)

                # 训练discriminator
                self.d_model.train()
                self.g_model.eval()
                self.d_optimizer.zero_grad()
                d_loss = 0.5 * (self.loss_fn(self.d_model(real_imgs), target_ones)) + \
                         0.5 * (self.loss_fn(self.d_model(fake_imgs.detach()), target_zeros))
                d_loss.backward()
                self.d_optimizer.step()

                # 训练generator
                self.d_model.eval()
                self.g_model.train()
                self.g_optimizer.zero_grad()
                g_loss = self.loss_fn(self.d_model(fake_imgs), target_ones)
                g_loss.backward()
                self.g_optimizer.step()

                pbar.update(1)

            tqdm.write(f"d_loss: {d_loss / self.loader.batch_size:>5f} || "
                       f"g_loss: {g_loss / self.loader.batch_size:>5f}")

        return fake_imgs


def run(model: Model,
        epochs,
        save_model_path,
        save_pic_path):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not os.path.exists(save_pic_path):
        os.mkdir(save_pic_path)

    for i in range(epochs):
        print(f"----------Training the {i + 1} time---------")
        fake_imgs = model.train()
        if i % 10 == 0:
            save_image(fake_imgs, os.path.join(save_pic_path, f'{i + 1}.png'))
        if i % 40 == 0:
            torch.save(model.g_model.state_dict(),
                       os.path.join(save_model_path, f'gen_model_weights_{i + 1}.pth'))
            torch.save(model.d_model.state_dict(),
                       os.path.join(save_model_path, f'dis_model_weights_{i + 1}.pth'))


def main(data_root, epochs):
    batch_size = 64

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    loader = MyLoader(data_root, batch_size)
    model = Model(loader, device)

    save_model_path = 'model'
    save_pic_path = 'pics'

    run(model, epochs, save_model_path, save_pic_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        type=str,
                        default=r'E:\desktop\mnist',
                        help="where the dataset is")

    parser.add_argument('--epochs',
                        type=int,
                        default=100)

    args = parser.parse_args()

    main(args.data_root, args.epochs)
