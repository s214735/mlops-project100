import torch
import hydra
from pytorch_lightning import LightningModule
from torch import nn
from torchvision.models import resnet50, resnet18
from torchmetrics.classification import Accuracy
from omegaconf import DictConfig

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    num_classes = cfg.model.num_classes
    lr = cfg.data.lr
    ResNetModel(num_classes, lr)

class ResNetModel(LightningModule):
    """A Lightning Module using ResNet-18 as the backbone."""

    def __init__(self, num_classes: int = 1000, lr: float = 0.001) -> None:
        super().__init__()

        # Load a pretrained ResNet-18 model
        self.backbone = resnet18(weights="ResNet18_Weights.DEFAULT")

        # Replace the final fully connected layer to match the number of classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Loss function
        self.criterium = nn.CrossEntropyLoss()

        # Hyperparameters
        self.lr = lr

        # Accuracies
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        data, target, _ = batch  # Adjust if your dataset returns additional items
        preds = self(data)
        loss = self.criterium(preds, target)

        acc = self.train_accuracy(preds, target)
        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch
        preds = self(data)
        loss = self.criterium(preds, target)

        acc = self.val_accuracy(preds, target)
        self.log("val_loss", loss)
        self.log("val_acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]



if __name__ == "__main__":
    model = ResNetModel(num_classes=1000)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
