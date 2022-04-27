import random

import numpy as np
import torch
from PIL import Image


def de_normalize_im(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.permute((1, 2, 0))
    mean = torch.Tensor([0.5, 0.5, 0.5]).to(tensor.device)
    std = torch.Tensor([0.5, 0.5, 0.5]).to(tensor.device)
    tensor = tensor * std + mean
    tensor = tensor * 255
    im_num_clip = np.clip(tensor.detach().cpu().numpy(), 0, 255)
    im_np = im_num_clip.astype(np.uint8)
    return im_np


def thumbnail(raw_im: np.ndarray, masked_im: np.ndarray, generated_im: np.ndarray) -> Image:
    concat = np.concatenate([raw_im, masked_im, generated_im], axis=1)
    image = Image.fromarray(concat)
    return image


def save_infer_sample(b_raw_im: torch.Tensor, b_masked_im: torch.Tensor, b_generated_im: torch.Tensor, save_path: str):
    num_samples = len(b_raw_im)

    idx = random.choice(range(num_samples))

    raw_im = de_normalize_im(b_raw_im[idx])
    masked_im = de_normalize_im(b_masked_im[idx])
    generated_im = de_normalize_im(b_generated_im[idx])

    image = thumbnail(raw_im, masked_im, generated_im)
    image.save(save_path)
