import torch
from utils import DiscriminatorLoss, GeneratorLoss
from torchvision import transforms
from model import Generator, Discriminator
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import argparse
import sys
from dataset import MnistDataset, NoiseDataset


class MyLoader:
    def __init__(self, real_data_root, batch_size):
        self.real_data_root = real_data_root
        self.batch_size = batch_size
        self.real_loader, self.noise_loader = self.get_dataloader()

    def get_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.1307], [0.3081])])
        real_data = MnistDataset(data_root=self.real_data_root, transform=transform)
        real_loader = DataLoader(real_data,
                                 batch_size=self.batch_size,
                                 drop_last=True)
        print('real data has been loaded over!!')

        noise_data = NoiseDataset(0, 1, 70000)
        noise_loader = DataLoader(noise_data,
                                  batch_size=self.batch_size,
                                  drop_last=True)
        print('noise data has been loaded over!!')

        return real_loader, noise_loader


class Model:
    def __init__(self, loader: MyLoader, device):
        self.loader = loader
        self.device = device
        self.lr = 2e-3

        self.g_model = Generator(100).to(device)
        self.g_loss_fn = GeneratorLoss()
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=self.lr)

        self.d_model = Discriminator(28 * 28).to(device)
        self.d_loss_fn = DiscriminatorLoss()
        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=self.lr)

    def train_discriminator(self):
        num_batches = len(self.loader.real_loader)
        self.d_model.train()
        self.g_model.eval()
        with tqdm(total=num_batches) as pbar:
            for _, (real, noise) in enumerate(zip(self.loader.real_loader, self.loader.noise_loader)):
                real, noise = real.to(self.device), noise.to(self.device)

                gen = self.g_model(noise)
                gen_img_score = self.d_model(gen)
                real_img_score = self.d_model(real)
                loss = self.d_loss_fn(real_img_score, gen_img_score)

                # Backpropagation
                self.d_optimizer.zero_grad()
                loss.backward()
                self.d_optimizer.step()
                pbar.update(1)
                pbar.set_postfix(loss=f'{loss.item() / self.loader.batch_size:>5f}')

            loss = loss.item()
            tqdm.write(f"discriminator loss: {loss / self.loader.batch_size:>7f}")

    def train_generator(self):
        num_batches = len(self.loader.noise_loader)
        self.g_model.train()
        with tqdm(total=num_batches) as pbar:
            for _, noise in enumerate(self.loader.noise_loader):
                noise = noise.to(self.device)

                # Compute prediction error
                gen = self.g_model(noise)
                gen_img_score = self.d_model(gen)
                loss = self.g_loss_fn(gen_img_score)
                # Backpropagation
                self.g_optimizer.zero_grad()
                loss.backward()
                self.g_optimizer.step()
                pbar.update(1)
                pbar.set_postfix(loss=f'{loss.item() / self.loader.batch_size:>5f}')

            loss = loss.item()
            tqdm.write(f"generator loss: {loss / self.loader.batch_size:>7f}")


def run(model: Model,
        t_epochs,
        save_model_path):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    for t in range(t_epochs):
        print(f"----------Training the {t + 1} time---------")
        print("training discriminator")
        for d in range(5):
            model.train_discriminator()
        print("\ntraining generator")
        for g in range(3):
            model.train_generator()
        print('\n\n')

        if t % 20 == 0:
            torch.save(model.g_model.state_dict(),
                       os.path.join(save_model_path, f'gen_model_weights_{t + 1}.pth'))
            torch.save(model.d_model.state_dict(),
                       os.path.join(save_model_path, f'dis_model_weights_{t + 1}.pth'))


def main(data_root, t_epochs):
    batch_size = 32

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    loader = MyLoader(data_root, batch_size)
    model = Model(loader, device)

    save_model_path = 'model'

    run(model, t_epochs, save_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        type=str,
                        default=r'E:\desktop\mnist',
                        help="where the dataset is")

    parser.add_argument('--t_epochs',
                        type=int,
                        default=100)

    args = parser.parse_args()

    main(args.data_root, args.t_epochs)
