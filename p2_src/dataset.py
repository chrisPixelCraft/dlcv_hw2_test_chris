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


class P2Dataset(Dataset):
    def __init__(self, pt_dir, transform=None):
        self.pt_dir = pt_dir
        self.transform = transform

        self.pt_files = sorted(
            [
                os.path.join(self.pt_dir, f)
                for f in os.listdir(self.pt_dir)
                if os.path.isfile(os.path.join(self.pt_dir, f))
            ]
        )

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):

        pt_file = self.pt_files[idx]

        pt = torch.load(pt_file)

        if self.transform:
            pass
            # pt = self.transform(pt)

        pt = pt.squeeze(0)
        return pt


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    noise_dir = os.path.join(parent_dir, "hw2_data/face/noise")
    gt_dir = os.path.join(parent_dir, "hw2_data/face/GT")

    img_dataset = P2Dataset(noise_dir, gt_dir, transform=transforms.ToTensor())

    dataloader = DataLoader(img_dataset, batch_size=2, shuffle=False, drop_last=True)
    print(len(dataloader))
    for i, (pt, gt) in enumerate(dataloader):
        print(pt.shape, gt.shape)
        break
