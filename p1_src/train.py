from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
import torchvision.transforms as trns
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import pandas as pd
from PIL import Image
import csv
from dataset import P1Dataset
from model import ContextUnet
from diffusion import DDPM


def train():
    # hardcoding these here
    n_epoch = 20
    batch_size = 256
    n_T = 400  # 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = 20
    n_feat = 256  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "P1/output/")
    ws_test = [0.0, 4.0, 10.0]  # strength of generative guidance

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
    )
    checkpoint = torch.load("best_p1_model.pth")
    ddpm.load_state_dict(checkpoint)
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor()]
    )  # mnist is already normalised 0 to 1

    # train_dir = "hw2_data/digits/mnistm/data"
    # train_dir_csv = "hw2_data/digits/mnistm/train.csv"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    train_dir_mnistm = os.path.join(parent_dir, "hw2_data/digits/mnistm/data")
    train_csv_mnistm = os.path.join(parent_dir, "hw2_data/digits/mnistm/train.csv")
    train_dir_svhn = os.path.join(parent_dir, "hw2_data/digits/svhn/data")
    train_csv_svhn = os.path.join(parent_dir, "hw2_data/digits/svhn/train.csv")

    train_dataset_mnistm = P1Dataset(
        csv_file=train_csv_mnistm,
        img_dir=train_dir_mnistm,
        transform=tf,
        dataset_name="mnistm",
    )
    train_dataset_svhn = P1Dataset(
        csv_file=train_csv_svhn,
        img_dir=train_dir_svhn,
        transform=tf,
        dataset_name="svhn",
    )

    train_dataset_combined = ConcatDataset([train_dataset_mnistm, train_dataset_svhn])

    dataloader = DataLoader(
        train_dataset_combined, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # dataset = ImageDataset(
    #     file_path=train_dir,
    #     csv_path=train_dir_csv,
    #     transform=tf,
    # )

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.AdamW(ddpm.parameters(), lr=lrate)
    # save_dir = "/content/P1/output/"
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(n_epoch):
        print(f"epoch {ep}")
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4 * n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(
                    n_sample, (3, 28, 28), device, guide_w=w
                )
                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample / n_classes)):
                        try:
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k + (j * n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all * -1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print("saved image at " + save_dir + f"image_ep{ep}_w{w}.png")

        # optionally save model
        if save_model and ep == int(n_epoch - 1) or ep % 10 == 0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print("saved model at " + save_dir + f"model_{ep}.pth")


if __name__ == "__main__":
    train()
