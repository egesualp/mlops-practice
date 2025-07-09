import pytorch_lightning as pl
import torch
from torch import nn
import wandb


class MyAwesomeButLightModel(pl.LightningModule):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        #self.save_hyperparameters() # make sure it accepts args via init
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.logger.experiment.log({'logits': wandb.Histogram(preds)})
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def validation_step(self, batch):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=1)).float().mean()
        self.log('val_loss', loss, on_epoch=True) # on_epoch = True because we want to log the val. loss only once per epoch
        self.log('val_acc', acc, on_epoch=True)
        # we don't return anything from val step



if __name__ == "__main__":
    model = MyAwesomeButLightModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
