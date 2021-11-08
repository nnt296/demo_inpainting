import os
from argparse import ArgumentParser

import torch

from unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main(h_params):
    model = Unet(h_params)

    os.makedirs(h_params.log_dir, exist_ok=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = [model_checkpoint, lr_monitor]

    ckpt_path = h_params.resume if len(h_params.resume) > 0 else None

    trainer = Trainer(
        max_epochs=h_params.max_epochs,
        gpus=1,
        callbacks=callbacks,
        num_nodes=1,
        ckpt_path=ckpt_path
    )

    trainer.fit(model)


if __name__ == '__main__':
    # For reproducible
    torch.manual_seed(123)

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='lightning_logs')

    parser = Unet.add_model_specific_args(parent_parser)
    params = parser.parse_args()

    main(params)
