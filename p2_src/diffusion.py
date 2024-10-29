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
from utils import beta_scheduler


def ddim_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    beta_t = beta_scheduler(T, beta1, beta2)
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    alphabar_t = torch.cumprod(alpha_t, dim=0)
    alpha_bar_prev = F.pad(alphabar_t[:-1], (1, 0), value=1.0)

    # forward calculations
    sqrt_alphas_cumprod = torch.sqrt(alphabar_t)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphabar_t)
    log_one_minus_alphas_cumprod = torch.log(1.0 - alphabar_t)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphabar_t)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphabar_t - 1)

    posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alphabar_t)
    posterior_log_variance_clipped = torch.log(
        torch.cat([posterior_variance[1:2], posterior_variance[1:]])
    )
    posterior_mean_coef1 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alphabar_t)
    posterior_mean_coef2 = (
        (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alphabar_t)
    )

    # log_alpha_t = torch.log(alpha_t)
    # alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    # sqrtab = torch.sqrt(alphabar_t)
    # oneover_sqrta = 1 / torch.sqrt(alpha_t)

    # sqrtmab = torch.sqrt(1 - alphabar_t)
    # mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "alphabar_t": alphabar_t,  # \bar{\alpha}_t
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,  # \sqrt{\bar{\alpha}_t}
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,  # \sqrt{1-\bar{\alpha}_t}
        "log_one_minus_alphas_cumprod": log_one_minus_alphas_cumprod,  # \log(1-\bar{\alpha}_t)
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,  # \sqrt{\frac{1}{\bar{\alpha}_t}}
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,  # \sqrt{\frac{1}{\bar{\alpha}_t} - 1}
        "posterior_variance": posterior_variance,  # \hat{\beta}_t
        "posterior_log_variance_clipped": posterior_log_variance_clipped,  # \log(\hat{\beta}_t)
        "posterior_mean_coef1": posterior_mean_coef1,  # \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}}}{\bar{\alpha}_t}
        "posterior_mean_coef2": posterior_mean_coef2,  # \frac{(1-\alpha_{t-1})\sqrt{\alpha_t}}{\bar{\alpha}_t}
    }


class DDIM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDIM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddim_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # slerp
    def slerp(self, alpha, v0, v1):
        # Ensure alpha is a column vector
        alpha = alpha.view(-1, 1, 1, 1)

        # Normalize the input vectors
        v0_norm = F.normalize(v0, dim=1)
        v1_norm = F.normalize(v1, dim=1)

        # Compute the cosine of the angle between the vectors
        omega = torch.acos((v0_norm * v1_norm).sum(dim=1, keepdim=True).clamp(-1, 1))
        so = torch.sin(omega)

        # Avoid division by zero
        eps = 1e-8
        so = torch.where(so.abs() < eps, torch.ones_like(so), so)

        # Perform SLERP
        return (torch.sin((1.0 - alpha) * omega) / so) * v0 + (
            torch.sin(alpha * omega) / so
        ) * v1

    # linear interpolation
    def linear_interpolation(self, pt0, pt1, batch_size):
        alphas = torch.linspace(0, 1, batch_size).to(pt0.device)
        return torch.lerp(pt0.unsqueeze(0), pt1.unsqueeze(0), alphas.view(-1, 1, 1, 1))

    def sample(
        self, pt, device, batch_size=10, ddim_timesteps=50, eta=0.0, clip_denoised=True
    ):
        # x_T ~ N(0, 1), sample initial noise
        x_i = pt.to(device)
        # Implement uniform time-step scheduler
        steps = ddim_timesteps
        c = self.n_T // steps
        ddim_timestep_seq = np.asarray(list(range(0, self.n_T, c)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        for i in tqdm(
            reversed(range(0, ddim_timesteps)),
            desc="Sampling loop time step",
            total=ddim_timesteps,
        ):
            t = torch.full(
                (batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long
            )
            t_prev = torch.full(
                (batch_size,),
                ddim_timestep_prev_seq[i],
                device=device,
                dtype=torch.long,
            )

            # get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphabar_t, t, x_i.shape)
            alpha_cumprod_t_prev = self._extract(self.alphabar_t, t_prev, x_i.shape)

            # predict noise using model
            pred_noise = self.nn_model(x_i, t)

            # get noise prediction
            pred_x0 = (x_i - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(
                alpha_cumprod_t
            )

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

            # compute variance
            sigmas_t = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev)
                / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            pred_dir_xt = (
                torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            )

            x_i = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0
                + pred_dir_xt
                + sigmas_t * torch.randn_like(x_i)
            )

        return x_i

    def slerp_linear_sample(
        self,
        pt0,
        pt1,
        device,
        batch_size=10,
        ddim_timesteps=50,
        eta=0.0,
        clip_denoised=True,
        interpolation_type="slerp",
    ):
        # Reshape pt0 and pt1 to remove the extra dimension
        pt0 = pt0.squeeze(0).to(device)
        pt1 = pt1.squeeze(0).to(device)
        alpha = torch.linspace(0, 1, batch_size, device=device)

        # Interpolate between pt0 and pt1
        if interpolation_type == "slerp":
            x_i = self.slerp(alpha, pt0, pt1)
        elif interpolation_type == "linear":
            x_i = self.linear_interpolation(pt0, pt1, batch_size)
        else:
            raise ValueError("Invalid interpolation type. Choose 'slerp' or 'linear'.")

        # Implement uniform time-step scheduler
        steps = ddim_timesteps
        c = self.n_T // steps
        ddim_timestep_seq = np.asarray(list(range(0, self.n_T, c)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        for i in tqdm(
            reversed(range(0, ddim_timesteps)),
            desc="Sampling loop time step",
            total=ddim_timesteps,
        ):
            t = torch.full(
                (batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long
            )
            t_prev = torch.full(
                (batch_size,),
                ddim_timestep_prev_seq[i],
                device=device,
                dtype=torch.long,
            )

            # get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphabar_t, t, x_i.shape)
            alpha_cumprod_t_prev = self._extract(self.alphabar_t, t_prev, x_i.shape)

            # predict noise using model
            pred_noise = self.nn_model(x_i, t)

            # get noise prediction
            pred_x0 = (x_i - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(
                alpha_cumprod_t
            )

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

            # compute variance
            sigmas_t = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev)
                / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            pred_dir_xt = (
                torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            )

            x_i = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0
                + pred_dir_xt
                + sigmas_t * torch.randn_like(x_i)
            )

        return x_i
