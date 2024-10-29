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
from model import ContextUnet
from diffusion import DDPM
from dataset import P1Dataset
import sys


def save_img(x_i, img_path, idx):
    cnt = 0
    tmp = 0
    for i in x_i:
        # print("len x_i", len(x_i))
        folder_name = "svhn" if tmp // 10 else "mnistm"
        label_folder = os.path.join(img_path, folder_name)
        path = os.path.join(label_folder, f"{cnt}_{str(idx+1).zfill(3)}.png")
        save_image(i, path, normalize=True, value_range=(-1, 1))
        # print(f"Saved image to {path}")
        cnt += 1
        tmp += 1
        cnt %= 10


def save_grid_image(img_dir, dataset_name):
    # Create a 10x10 grid
    grid = [[None for _ in range(10)] for _ in range(10)]

    # Iterate through files in the directory
    pbar = tqdm(sorted(os.listdir(img_dir)), desc=f"Saving {dataset_name} grid image")
    for filename in pbar:
        if filename.endswith(".png"):
            # Extract indices from filename
            row = int(filename.split("_")[0])
            col = int(filename.split("_")[1].split(".")[0]) - 1
            if col >= 10:
                continue
            # Load the image
            img_path = os.path.join(img_dir, filename)
            img = Image.open(img_path)

            # Place the image in the grid
            grid[row][col] = np.array(img)

    # Create a figure to display the grid
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    # Populate the figure with images
    for i in range(10):
        for j in range(10):
            if grid[i][j] is not None:
                axes[i, j].imshow(grid[i][j])
            axes[i, j].axis("off")

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_grid.png")
    plt.close()


def main():
    # Modify the output directories to use the command-line argument
    if len(sys.argv) > 1:
        output_base_dir = sys.argv[1]
    else:
        output_base_dir = os.path.join(current_dir, "Output_folder")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # hardcoding these here
    batch_size = 256
    n_T = 400  # 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = 20
    n_feat = 256  # 128 ok, 256 better (but slower)
    ws_test = [0.0, 4.0, 10.0]  # strength of generative guidance

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # checkpoint_dir = os.path.dirname(current_dir)
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

    images = []
    zero_image = []
    # image_dir = os.path.join(current_dir, "Output_folder")
    image_dir = output_base_dir
    image_dir_store = os.path.join(current_dir, "Denoising_Timesteps")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(image_dir_store, exist_ok=True)
    os.makedirs(os.path.join(image_dir, "mnistm"), exist_ok=True)
    os.makedirs(os.path.join(image_dir, "svhn"), exist_ok=True)

    with torch.no_grad():
        n_sample = 1 * n_classes
        w = 40.0
        pbar = tqdm(range(50), desc="Generating images")
        for i in pbar:
            # control sample 20 images for each class once, then loop 50 times
            x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=w)
            # use i as index for naming the images (save inference images)
            save_img(x_gen, image_dir, i)

            # save timestep images for label 0
            len_x_gen_store = len(x_gen_store)
            if i == 0:
                for j in range(len_x_gen_store):
                    # print("type of x_gen_store[j]: ", type(x_gen_store[j]))
                    # print(x_gen_store[j].shape)
                    img = torch.tensor(x_gen_store[j])
                    save_image(img, os.path.join(image_dir_store, f"timestep_{j}.png"))

            # combine 10 0-9 images into 10x10 grid
    print("Saving grid image...")
    mnistm_dir = os.path.join(image_dir, "mnistm")
    svhn_dir = os.path.join(image_dir, "svhn")
    save_grid_image(mnistm_dir, "mnistm")
    save_grid_image(svhn_dir, "svhn")


if __name__ == "__main__":
    main()
