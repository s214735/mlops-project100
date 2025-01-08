import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision.models import resnet50


class ResNetModel(LightningModule):
    """A Lightning Module using ResNet-50 as the backbone."""

    def __init__(self, num_classes: int, lr: float) -> None:
        super().__init__()

        # Load a pretrained ResNet-50 model
        self.backbone = resnet50(weights="ResNet50_Weights.DEFAULT")

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
        data, target, _ = batch  # Adjust if your dataset returns additional items
        preds = self(data)
        loss = self.criterium(preds, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]



if __name__ == "__main__":
    model = ResNetModel(num_classes=1000)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
