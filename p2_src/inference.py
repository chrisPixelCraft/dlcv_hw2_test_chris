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

# from model import ContextUnet
from diffusion import DDIM

from dataset import P2Dataset
from model import UNet
import sys


def save_img(x_i, img_path, idx):
    cnt = 0
    for i in x_i:
        path = os.path.join(img_path, f"{str(cnt).zfill(2)}.png")
        save_image(i, path, normalize=True, value_range=(-1, 1))
        cnt += 1


def save_img_grid(x_i, img_path, idx, eta):
    cnt = 0
    for i in x_i:
        path = os.path.join(img_path, f"{str(cnt).zfill(2)}_{eta}.png")
        save_image(i, path, normalize=True, value_range=(-1, 1))
        cnt += 1


def eta_img_to_grid(grid_dir):
    grid = [[None for _ in range(5)] for _ in range(5)]  # Changed to 5x5 grid
    pbar = tqdm(sorted(os.listdir(grid_dir)), desc="Saving eta grid image")
    for filename in pbar:
        if filename.endswith(".png"):
            col = int(filename.split("_")[0])
            row_value = float(filename.split("_")[1].split(".png")[0])
            # print("row_value", row_value)
            row = int(
                row_value * 4
            )  # This maps 0.0 to 0, 0.25 to 1, 0.50 to 2, 0.75 to 3, and 1.0 to 4
            # print("col", col)
            # print("row", row)
            if col >= 5:
                continue
            img_path = os.path.join(grid_dir, filename)
            img = Image.open(img_path)
            grid[row][col] = np.array(img)

    fig, axes = plt.subplots(5, 4, figsize=(20, 20))

    for i in range(5):
        for j in range(4):
            if grid[i][j] is not None:
                axes[i, j].imshow(grid[i][j])
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("p2_eta_grid.png")
    plt.close()


def interpolate_img_to_grid(img_dir, interpolation_type):
    grid = [[None for _ in range(10)] for _ in range(1)]

    i = 0
    pbar = tqdm(sorted(os.listdir(img_dir)), desc=f"Saving interpolated grid image")
    for filename in pbar:
        if filename.endswith(".png"):
            img_path = os.path.join(img_dir, filename)
            img = Image.open(img_path)
            grid[0][i] = np.array(img)
            i += 1

    fig, axes = plt.subplots(1, 10, figsize=(20, 20))

    for i in range(10):
        if grid[0][i] is not None:
            axes[i].imshow(grid[0][i])
            axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"p2_{interpolation_type}_grid.png")
    plt.close()


def normalize_tensor(tensor):
    # Ensure the tensor is in the range [0, 1]
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val)

    # Scale to [0, 255] and convert to uint8
    return (normalized * 255).to(torch.uint8)


def compare_mse(x_i_path, gt_path):
    x_i = Image.open(x_i_path)
    gt = Image.open(gt_path)

    # Convert PIL Images to tensors
    transform = transforms.ToTensor()
    x_i = transform(x_i)
    gt = transform(gt)

    # Normalize both x_i and gt to 0-255 range
    x_i = normalize_tensor(x_i)
    gt = normalize_tensor(gt)

    assert x_i.shape == gt.shape
    mse = torch.nn.functional.mse_loss(x_i.float(), gt.float())
    print(f"MSE for image pair: {mse.item():5f}")
    return mse.item()


# def main(sys_argv_1, sys_argv_2, sys_argv_3):
def main():
    # Modify the output directories to use the command-line argument
    # if len(sys.argv) > 1:
    #     output_base_dir = sys.argv[1]
    # else:
    #     output_base_dir = os.path.join(current_dir, "Output_folder")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # hardcoding these here
    batch_size = 10
    eta = 0.0
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # noise_dir = os.path.join(parent_dir, "hw2_data/face/noise")
    # gt_dir = os.path.join(parent_dir, "hw2_data/face/GT")
    noise_dir = sys.argv[1]
    # gt_dir = sys.argv[2]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    img_dataset = P2Dataset(noise_dir, transform=transforms.ToTensor())

    dataloader = DataLoader(
        img_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = UNet()
    checkpoint_path = sys.argv[3]
    # print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(sys.argv[3])
    model.load_state_dict(checkpoint)
    model.to(device)
    ddim = DDIM(
        nn_model=model,
        betas=(1e-4, 2e-2),
        n_T=n_T,
        device=device,
        drop_prob=0.0,
    )
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # checkpoint_dir = os.path.dirname(current_dir)

    # tf = transforms.Compose([transforms.ToTensor()])
    # output_dir = sys.argv[2]

    image_dir = os.path.join(output_dir)
    # grid_dir = os.path.join(current_dir, "grid_folder_p2")
    # slerp_dir = os.path.join(current_dir, "slerp_folder_p2")
    # linear_dir = os.path.join(current_dir, "linear_folder_p2")
    os.makedirs(image_dir, exist_ok=True)
    # os.makedirs(grid_dir, exist_ok=True)
    # os.makedirs(slerp_dir, exist_ok=True)
    # os.makedirs(linear_dir, exist_ok=True)
    # image_dir = sys.argv[1]

    with torch.no_grad():
        for i, pt in enumerate(dataloader):
            # print(f"Processing batch {i+1}")
            pt = pt.to(device)
            # gt = gt.to(device)
            x_i = ddim.sample(pt, device, batch_size, eta=0.0)
            save_img(x_i, image_dir, i)

    # img = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    # gt = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])
    # mse = 0.0
    # for image, ground_truth in zip(img, gt):
    #     # print(image, ground_truth)
    #     path_image = os.path.join(image_dir, image)
    #     path_ground_truth = os.path.join(gt_dir, ground_truth)
    #     mse += compare_mse(path_image, path_ground_truth)
    # avg_mse = mse / len(img)
    # print(f"Average MSE: {avg_mse}")

    # # generate for different eta
    # with torch.no_grad():
    #     for eta in np.arange(0.0, 1.25, 0.25):
    #         for i, (pt, gt) in enumerate(dataloader):
    #             # print(f"Processing batch {i+1}")
    #             pt = pt.to(device)
    #             gt = gt.to(device)
    #             x_i = ddim.sample(pt, device, batch_size, eta=eta)
    #             save_img_grid(x_i, grid_dir, i, eta)

    # # save grid images
    # eta_img_to_grid(grid_dir)

    # # slerp and linear interpolation
    # with torch.no_grad():
    #     # Load 00.pt and 01.pt
    #     pt0 = torch.load(os.path.join(noise_dir, "00.pt"))
    #     pt1 = torch.load(os.path.join(noise_dir, "01.pt"))

    #     print(f"Processing SLERP between 00.pt and 01.pt")
    #     x_i = ddim.slerp_linear_sample(
    #         pt0, pt1, device, batch_size, eta=0.0, interpolation_type="slerp"
    #     )
    #     save_img(x_i, slerp_dir, "slerp_00_01")

    #     print(f"Processing Linear Interpolation between 00.pt and 01.pt")
    #     x_i = ddim.slerp_linear_sample(
    #         pt0, pt1, device, batch_size, eta=0.0, interpolation_type="linear"
    #     )
    #     save_img(x_i, linear_dir, "linear_00_01")

    # # save grid images
    # interpolate_img_to_grid(linear_dir, "linear")
    # interpolate_img_to_grid(slerp_dir, "slerp")


if __name__ == "__main__":
    # main(sys.argv[1])
    # main(sys.argv[1], sys.argv[2], sys.argv[3])
    main()
