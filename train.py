import torch
from utils import DiscriminatorLoss, GeneratorLoss
from torchvision import transforms
from model import Generator, Discriminator
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from dataset import MnistDataset, NoiseDataset



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
            real_img_score = d_model(real)
            loss = loss_fn(real_img_score, gen_img_score)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
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

    noise_data = NoiseDataset(0, 1, 70000, [28, 28])
    noise_loader = DataLoader(noise_data,
                              batch_size=batch_size,
                              drop_last=True,
                              pin_memory=True)
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


def main():
    gen_model = Generator((28, 28))
    gen_loss_fn = GeneratorLoss()
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=1e-3)

    dis_model = Discriminator()
    dis_loss_fn = DiscriminatorLoss()
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=1e-3)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    real_loader, noise_loader = get_dataloader('E:\desktop\mnist', 32)
    save_model_path = 'model'

    run(gen_model, gen_loss_fn, gen_optimizer,
        dis_model, dis_loss_fn, dis_optimizer,
        real_loader, noise_loader,
        30, 30, 5,
        save_model_path,
        device)


if __name__ == '__main__':
    main()