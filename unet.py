import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers import linear_warmup_decay
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from dataset import FoodDataset
from utils import save_infer_sample


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
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
                                         kernel_size=(2, 2), stride=(2, 2))

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [?, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)  # why 1?
        return self.conv(x)


class Unet(pl.LightningModule):
    def __init__(self, h_params):
        # TODO, use variables instead of h_params, so that Trainer can save hyper-params to files

        super(Unet, self).__init__()
        self.h_params = h_params

        self.n_channels = h_params.n_channels
        self.n_classes = h_params.n_classes
        self.bilinear = True

        # Do not do reduction
        self.mse_mean = torch.nn.MSELoss(reduction="mean")
        self.mse_none = torch.nn.MSELoss(reduction="none")

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=(1, 1))

        # Keep track of last loss
        self.last_loss = float("inf")

        self.train_ds = FoodDataset(img_dir=os.path.join(self.h_params.dataset, "images"),
                                    im_names_path=os.path.join(self.h_params.dataset, "meta", "meta", "train.txt"))
        self.num_samples = len(self.train_ds)
        self.train_iters_per_epoch = self.num_samples // self.h_params.batch_size

        self.val_ds = FoodDataset(img_dir=os.path.join(self.h_params.dataset, "images"),
                                  im_names_path=os.path.join(self.h_params.dataset, "meta", "meta", "test.txt"))
        self.viz_iters = min(len(self.val_ds), self.h_params.viz_iters)

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
        b_generated_im = self(b_masked_im)

        loss = self.mse_none(b_generated_im, b_raw_im)
        loss = loss.reshape(loss.size(0), -1)
        loss = torch.sum(loss, dim=-1)
        loss = loss / b_num_pixels
        loss = torch.mean(loss)

        # TODO: add more weight to masked pixels
        # https://stackoverflow.com/questions/61580037/mseloss-when-mask-is-used

        return loss, b_generated_im

    def training_step(self, batch, batch_nb):
        loss, _ = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, b_generated_im = self.shared_step(batch)

        # Random save image to view log
        if batch_nb % self.viz_iters == 0:
            b_raw_im, b_masked_im, _ = batch
            save_path = os.path.join(self.h_params.log_dir, f"output_epoch{self.current_epoch:03d}_batch{batch_nb}.png")
            save_infer_sample(b_raw_im, b_masked_im, b_generated_im, save_path)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.h_params.learning_rate, weight_decay=self.h_params.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.h_params.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.h_params.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            num_workers=self.h_params.num_workers,
            batch_size=self.h_params.batch_size,
            shuffle=True,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            num_workers=self.h_params.num_workers,
            batch_size=self.h_params.batch_size,
            shuffle=False,
            drop_last=True
        )
        return val_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # Model params
        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=3)

        # Training params
        parser.add_argument('--resume', type=str, default="")
        parser.add_argument('--max_epochs', type=int, default=50)
        parser.add_argument('--warmup_epochs', type=int, default=1)
        parser.add_argument('--reduction_point', type=float, default=0.05)
        parser.add_argument('--weight_decay', type=float, default=1e-6)
        parser.add_argument('--viz_iters', type=int, default=500)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        return parser


if __name__ == '__main__':
    p = ArgumentParser(add_help=False)
    p = Unet.add_model_specific_args(p)
    args = p.parse_args()
    net = Unet(args)

    im = torch.rand((2, 3, 512, 512))
    out = net(im)
    print(out.shape)
