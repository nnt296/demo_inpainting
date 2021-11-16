from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

from unet import Unet
from utils import de_normalize_im, thumbnail


def infer(image_path: str, model, de_noise=False, use_cuda: bool = True):
    raw = Image.open(image_path).convert("RGB")
    im = ImageEnhance.Contrast(raw).enhance(0.628)

    trans = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    im = trans(im).unsqueeze(0)
    if use_cuda:
        im = im.cuda()
    with torch.no_grad():
        out = model(im)

    raw_im = de_normalize_im(im[0])
    raw_im = Image.fromarray(raw_im)
    raw_im = ImageEnhance.Contrast(raw_im).enhance(1. / 0.628)
    raw_im = np.array(raw_im)

    generated_im = de_normalize_im(out[0])

    if de_noise:
        cv_im = cv2.cvtColor(generated_im, cv2.COLOR_RGB2BGR)
        de_noised_im = cv2.fastNlMeansDenoisingColored(cv_im, None, 3, 3, 5, 7)
        generated_im = cv2.cvtColor(de_noised_im, cv2.COLOR_BGR2RGB)

    # Enhance generated image
    generated_im = Image.fromarray(generated_im)
    generated_im = ImageEnhance.Contrast(generated_im).enhance(1. / 0.628)
    generated_im.save("output/gen.png")

    generated_im = np.array(generated_im)

    print(raw_im.shape, generated_im.shape)

    third = np.zeros_like(raw_im)
    image = thumbnail(raw_im, generated_im, third)
    image.save("output/output.png")


def save_model(h_params, save_for_gpu=True):
    ckpt_path = h_params.resume if len(h_params.resume) > 0 else "epoch.ckpt"
    model = Unet.load_from_checkpoint(ckpt_path, **h_params.__dict__)
    model.eval()

    if save_for_gpu:
        model.cuda()
        sample_inp = torch.randn((1, 3, 256, 256)).cuda()
        traced = torch.jit.trace(model, sample_inp)
        torch.jit.save(traced, "model_gpu.pt")
    else:
        model.cpu()
        sample_inp = torch.randn((1, 3, 256, 256))
        traced = torch.jit.trace(model, sample_inp)
        torch.jit.save(traced, "model_cpu.pt")


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--image', type=str, required=True)
    # parent_parser.add_argument('--dataset', type=str, required=True)
    parent_parser.add_argument('--cuda', action="store_true")
    parser = Unet.add_model_specific_args(parent_parser)
    params = parser.parse_args()

    net = torch.jit.load("model_gpu.pt")
    net.eval()

    # save_model(h_params=params, save_for_gpu=params.cuda)
    infer(params.image, model=net, de_noise=True, use_cuda=params.cuda)
