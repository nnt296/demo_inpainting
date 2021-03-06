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
    def __init__(
            self,
            dataset: str,
            n_channels: int = 3,
            n_classes: int = 3,
            bilinear: bool = False,

            learning_rate: float = 1e-3,
            batch_size: int = 4,
            log_dir: str = "lightning_logs",
            weight_decay: float = 1e-5,
            warmup_epochs: int = 1,
            max_epochs: int = 50,
            num_workers: int = 4,
            viz_iters: int = 100,
            **kwargs
    ):
        self.save_hyperparameters()

        super(Unet, self).__init__()
        self.dataset_dir = dataset

        # Model params
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = Up(1024, 256, bilinear=self.bilinear)
        self.up2 = Up(512, 128, bilinear=self.bilinear)
        self.dropout2 = nn.Dropout(p=0.2)
        self.up3 = Up(256, 64, bilinear=self.bilinear)
        self.up4 = Up(128, 64, bilinear=self.bilinear)
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=(1, 1))

        # Dataset and Loss
        self.mse_mean = torch.nn.MSELoss(reduction="mean")
        self.mse_none = torch.nn.MSELoss(reduction="none")

        # Training params
        self.num_train_samples = kwargs.get("num_samples", 100)
        self.num_val_samples = kwargs.get("num_val_samples", 100)
        self.batch_size = batch_size
        self.train_iters_per_epoch = self.num_train_samples // batch_size
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.viz_iters = min(self.num_val_samples, viz_iters)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.dropout1(x3)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.dropout2(x)
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
            save_path = os.path.join(self.log_dir, f"output_epoch{self.current_epoch:03d}_batch{batch_nb}.png")
            save_infer_sample(b_raw_im, b_masked_im, b_generated_im, save_path)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

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
        parser.add_argument('--bilinear', action="store_true")
        parser.add_argument('--reduction_point', type=float, default=0.05)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--viz_iters', type=int, default=500)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        return parser


if __name__ == '__main__':
    p = ArgumentParser(add_help=False)
    p = Unet.add_model_specific_args(p)
    args = p.parse_args()
    net = Unet(**args.__dict__)

    im = torch.rand((2, 3, 512, 512))
    out = net(im)
    print(out.shape)
