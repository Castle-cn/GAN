import torch
from utils import DiscriminatorLoss, GeneratorLoss
from torchvision import transforms
from model import Generator, Discriminator
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import argparse
from dataset import MnistDataset, NoiseDataset


class MyLoader:
    def __init__(self, real_data_root, batch_size, fake_img_size: list):
        self.real_data_root = real_data_root
        self.batch_size = batch_size
        self.fake_img_size = fake_img_size
        self.real_loader, self.noise_loader = self.get_dataloader()

    def get_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.1307], [0.3081])])
        real_data = MnistDataset(data_root=self.real_data_root, transform=transform)
        real_loader = DataLoader(real_data,
                                 batch_size=self.batch_size,
                                 drop_last=True)
        print('real data has been loaded over!!')

        noise_data = NoiseDataset(0, 1, 10000, [64, 64])
        noise_loader = DataLoader(noise_data,
                                  batch_size=self.batch_size,
                                  drop_last=True)
        print('noise data has been loaded over!!')

        return real_loader, noise_loader

    # def get_gen_dataloader(self, gen_model, device):
    #     gen_model.eval()
    #     gen_images = []
    #     num_batches = len(self.noise_loader)
    #
    #     tqdm.write("generating fake images!")
    #     with tqdm(total=num_batches) as pbar:
    #         for _, noise in enumerate(self.noise_loader):
    #             noise = noise.to(device)
    #             gen_image = gen_model(noise)
    #             gen_images.append(gen_image)
    #             pbar.update(1)
    #
    #     gen_images = torch.cat(gen_images, dim=0)
    #     gen_data = GenImage(gen_images, self.fake_img_size)
    #     gen_dataloader = DataLoader(gen_data,
    #                                 batch_size=self.batch_size,
    #                                 drop_last=True)
    #     tqdm.write("generating over!\n")
    #
    #     return gen_dataloader


class Model:
    def __init__(self, loader: MyLoader, d_model, d_loss_fn, d_optimizer,
                 g_model, g_loss_fn, g_optimizer, device):
        self.loader = loader
        self.device = device

        self.d_model = d_model.to(device)
        self.d_loss_fn = d_loss_fn
        self.d_optimizer = d_optimizer

        self.g_model = g_model.to(device)
        self.g_loss_fn = g_loss_fn
        self.g_optimizer = g_optimizer

    def train_discriminator(self):
        num_batches = len(self.loader.real_loader)
        self.d_model.train()
        with tqdm(total=num_batches) as pbar:
            for _, (real, noise) in enumerate(zip(self.loader.real_loader, self.loader.noise_loader)):
                real, noise = real.to(self.device), noise.to(self.device)

                gen = self.g_model(noise)
                # normalize the fake images
                mean = torch.mean(gen)
                std = torch.std(gen)
                gen = (gen - mean) / std

                gen_img_score = self.d_model(gen)
                real_img_score = self.d_model(real)
                loss = self.d_loss_fn(real_img_score, gen_img_score)

                # Backpropagation
                self.d_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.d_optimizer.step()
                pbar.update(1)
                pbar.set_postfix(loss=f'{loss.item():>5f}')

            loss = loss.item()
            tqdm.write(f"discriminator loss: {loss:>7f}")

    def train_generator(self):
        num_batches = len(self.loader.noise_loader)
        self.g_model.train()
        with tqdm(total=num_batches) as pbar:
            for batch, noise in enumerate(self.loader.noise_loader):
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
                pbar.set_postfix(loss=f'{loss.item():>5f}')

            loss = loss.item()
            tqdm.write(f"generator loss: {loss:>7f}")


def run(model: Model,
        t_epochs,
        d_epochs,
        g_epochs,
        save_model_path):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    for t in range(t_epochs):
        print(f"----------Training the {t + 1} time---------")
        print("training discriminator")
        # gen_loader = model.loader.get_gen_dataloader(model.g_model, model.device)
        for d in range(d_epochs):
            print(f"----------Epoch {d + 1} ---------")
            model.train_discriminator()
        print("\ntraining generator")
        for g in range(g_epochs):
            print(f"----------Epoch {g + 1} ---------")
            model.train_generator()
        print('\n\n')

        torch.save(model.g_model.state_dict(),
                   os.path.join(save_model_path, f'gen_model_weights_{t + 1}.pth'))
        torch.save(model.d_model.state_dict(),
                   os.path.join(save_model_path, f'dis_model_weights_{t + 1}.pth'))


def main(data_root, t_epochs, d_epochs, g_epochs):
    batch_size = 32
    lr = 1e-3

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    gen_model = Generator([28, 28], batch_size)
    gen_loss_fn = GeneratorLoss()
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=lr)

    dis_model = Discriminator()
    dis_loss_fn = DiscriminatorLoss()
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=lr)

    loader = MyLoader(data_root, batch_size, [28, 28])
    model = Model(loader, dis_model, dis_loss_fn, dis_optimizer,
                  gen_model, gen_loss_fn, gen_optimizer, device)

    save_model_path = 'model'

    run(model, t_epochs, d_epochs, g_epochs, save_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        type=str,
                        default=r'E:\desktop\mnist',
                        help="where the dataset is")

    parser.add_argument('--t_epochs',
                        type=int,
                        default=30)

    parser.add_argument('--d_epochs',
                        type=int,
                        default=5)

    parser.add_argument('--g_epochs',
                        type=int,
                        default=3)

    args = parser.parse_args()

    main(args.data_root, args.t_epochs, args.d_epochs, args.g_epochs)
