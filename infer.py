from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

from unet import Unet
from utils import de_normalize_im, thumbnail


def infer(h_params):
    ckpt_path = h_params.resume if len(h_params.resume) > 0 else None
    model = Unet.load_from_checkpoint(ckpt_path, **h_params.__dict__)
    model.eval()

    im = Image.open(h_params.image).convert("RGB")
    im = ImageEnhance.Contrast(im).enhance(0.628)

    trans = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    im = trans(im).unsqueeze(0)
    with torch.no_grad():
        out = model(im)

    raw_im = de_normalize_im(im[0])
    generated_im = de_normalize_im(out[0])
    third = np.zeros_like(raw_im)
    image = thumbnail(raw_im, generated_im, third)
    image.save("output/output.png")


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--image', type=str, required=True)
    parent_parser.add_argument('--dataset', type=str, required=True)
    parser = Unet.add_model_specific_args(parent_parser)
    params = parser.parse_args()

    infer(params)
