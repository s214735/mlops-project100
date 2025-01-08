
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torchvision.models import resnet50


class ResNetModel(pl.LightningModule):
    """A Lightning Module using ResNet-50 as the backbone."""

    def __init__(self, num_classes: int = 1000, lr: float = 1e-3):
        super().__init__()

        # Load a pretrained ResNet-50 model
        self.backbone = resnet50(weights='ResNet50_Weights.DEFAULT')

        # Replace the final fully connected layer to match the number of classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Loss function
        self.criterium = nn.CrossEntropyLoss()

        # Hyperparameters
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize the DataModule
    data_module = CustomDataModule(
        data_path=Path(cfg.dataset.processed_data_path),
        batch_size=cfg.dataset.batch_size
    )

    # Initialize the model
    model = ResNetModel(num_classes=cfg.model.num_classes, lr=cfg.optimizer.lr)

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision
    )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
