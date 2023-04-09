import torch
from utils import DiscriminatorLoss, GeneratorLoss
from torchvision import transforms
from model import Generator, Discriminator
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import argparse
from dataset import MnistDataset, NoiseDataset
import sys


# real_dataloader的batch_size和num_batch要和noise_dataloader一致
def train_discriminator(real_dataloader,
                        noise_dataloader,
                        d_model,
                        g_model,
                        loss_fn,
                        optimizer,
                        device):
    num_batches = len(real_dataloader)
    d_model.train()
    with tqdm(total=num_batches) as pbar:
        for batch, (real, noise) in enumerate(zip(real_dataloader, noise_dataloader)):
            real, noise = real.to(device), noise.to(device)

            # Compute prediction error
            gen = g_model(noise)
            gen_img_score = d_model(gen)
            # print(gen_img_score)
            real_img_score = d_model(real)
            # print(real_img_score)
            # sys.exit()
            loss = loss_fn(real_img_score, gen_img_score)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix(loss=f'{loss.item():>5f}')

        loss = loss.item()
        tqdm.write(f"discriminator loss: {loss:>7f}")


def train_generator(noise_dataloader,
                    d_model,
                    g_model,
                    loss_fn,
                    optimizer,
                    device):
    num_batches = len(noise_dataloader)
    g_model.train()
    with tqdm(total=num_batches) as pbar:
        for batch, noise in enumerate(noise_dataloader):
            noise = noise.to(device)

            # Compute prediction error
            gen = g_model(noise)
            gen_img_score = d_model(gen)
            loss = loss_fn(gen_img_score)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix(loss=f'{loss.item():>5f}')

        loss = loss.item()
        tqdm.write(f"generator loss: {loss:>7f}")


def get_dataloader(real_data_root, batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.1307], [0.3081])])
    real_data = MnistDataset(data_root=real_data_root, transform=transform)
    real_loader = DataLoader(real_data,
                             batch_size=batch_size,
                             drop_last=True,
                             pin_memory=True)
    print('real data has been loaded over!!')

    noise_data = NoiseDataset(0, 1, 70000, [64, 64])
    noise_loader = DataLoader(noise_data,
                              batch_size=batch_size,
                              drop_last=True,
                              pin_memory=True)
    print('noise data has been loaded over!!')

    return real_loader, noise_loader


def run(gen_model,
        gen_loss_fn,
        gen_optimizer,
        dis_model,
        dis_loss_fn,
        dis_optimizer,
        real_loader,
        noise_loader,
        t_epochs,
        d_epochs,
        g_epochs,
        save_model_path,
        device):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    for t in range(t_epochs):
        print("----------Training the {t + 1} time---------")
        print("training discriminator")
        for t in range(d_epochs):
            print(f"----------Epoch {t + 1} ---------")
            train_discriminator(real_loader, noise_loader, dis_model,
                                gen_model, dis_loss_fn, dis_optimizer, device)
        print('\n\n')
        print("training generator")
        for t in range(g_epochs):
            print(f"----------Epoch {t + 1} ---------")
            train_generator(noise_loader, dis_model, gen_model,
                            gen_loss_fn, gen_optimizer, device)
        print('\n\n')

        torch.save(gen_model.state_dict(),
                   os.path.join(save_model_path, f'gen_model_weights_{t + 1}.pth'))
        torch.save(dis_model.state_dict(),
                   os.path.join(save_model_path, f'dis_model_weights_{t + 1}.pth'))


def main(data_root):
    batch_size = 32
    lr = 1e-3

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    gen_model = Generator((28, 28), batch_size).to(device)
    gen_loss_fn = GeneratorLoss()
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=lr)

    dis_model = Discriminator().to(device)
    dis_loss_fn = DiscriminatorLoss()
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=lr)

    real_loader, noise_loader = get_dataloader(data_root, batch_size=batch_size)

    save_model_path = 'model'

    run(gen_model, gen_loss_fn, gen_optimizer,
        dis_model, dis_loss_fn, dis_optimizer,
        real_loader, noise_loader,
        30, 10, 2,
        save_model_path,
        device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        type=str,
                        required=True,
                        help="where the dataset is")
    args = parser.parse_args()

    main(args.data_root)
