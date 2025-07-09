from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

import matplotlib.pyplot as plt
import torch
import typer
from model_light import MyAwesomeButLightModel

from data import corrupt_mnist
from datetime import datetime
import os
import sys

#logger.remove()
#logger.add(sys.stdout, level='WARNING')

def train_light(lr: float = 1e-3, batch_size: int = 32, max_epochs: int = 10, max_steps: int = -1) -> None:
    """Train a model on MNIST."""

    if max_steps is None:
        max_steps = -1
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    model = MyAwesomeButLightModel()

    early_stopping = EarlyStopping(
        monitor='train_loss', patience=3, verbose=True, mode='min'
    )

    checkpoint = ModelCheckpoint(
        dirpath='./models', monitor='train_loss', mode='min'
    )

    trainer = Trainer(
        default_root_dir='.',
        max_epochs=max_epochs,
        max_steps=max_steps,
        logger=WandbLogger(project='mlops_light_mnist'),
        accelerator='auto',
        devices=1,
        limit_train_batches=0.2,
        callbacks=[early_stopping, checkpoint],
        profiler="simple"
    )

    trainer.fit(model, train_dataloaders=train_dataloader)

if __name__ == "__main__":
    typer.run(train_light)