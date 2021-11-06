import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from dataset import FoodDataset


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down(in_channels, out_channels):
    return nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_channels, out_channels)
    )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [?, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # why 1?
        return self.conv(x)


class Unet(pl.LightningModule):
    def __init__(self, h_params):
        super(Unet, self).__init__()
        self.h_params = h_params

        self.n_channels = h_params.n_channels
        self.n_classes = h_params.n_classes
        self.bilinear = True

        # Do not do reduction
        self.mse = torch.nn.MSELoss(reduction="none")

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

    def shared_step(self, batch):
        # final image in tuple is for online eval
        b_raw_im, b_masked_im, b_num_pixels = batch

        # get h representations, bolts resnet returns a list
        generated_im = self(b_masked_im)

        loss = self.mse(generated_im, b_raw_im)
        loss = loss / b_num_pixels

        return loss

    def training_step(self, batch, batch_nb):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(net.parameters(), lr=self.h_params.learning_rate)

    def train_dataloader(self):
        train_ds = FoodDataset(img_dir=os.path.join(self.h_params.dataset, "images"),
                               im_names_path=os.path.join(self.h_params.dataset, "meta", "meta", "train.txt"))

        train_loader = DataLoader(train_ds, batch_size=self.h_params.batch_size, shuffle=True, drop_last=True)
        return train_loader

    def val_dataloader(self):
        val_ds = FoodDataset(img_dir=os.path.join(self.h_params.dataset, "images"),
                             im_names_path=os.path.join(self.h_params.dataset, "meta", "meta", "test.txt"))

        val_loader = DataLoader(val_ds, batch_size=self.h_params.batch_size, shuffle=False, drop_last=True)
        return val_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--learning_rate', type=int, default=1e-3)
        return parser


if __name__ == '__main__':
    p = ArgumentParser(add_help=False)
    p = Unet.add_model_specific_args(p)
    params = p.parse_args()
    net = Unet(params)

    im = torch.rand((2, 3, 512, 512))
    out = net(im)
    print(out.shape)
