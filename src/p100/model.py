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
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = ResNetModel(num_classes=1000)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
