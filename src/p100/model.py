import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification import Accuracy
from torchvision.models import resnet50


class ResNetModel(LightningModule):
    """A Lightning Module using ResNet-50 as the backbone."""

    def __init__(self, num_classes: int = 1000, lr: float = 0.001) -> None:
        super().__init__()

        # Load a pretrained ResNet-50 model
        self.backbone = resnet50(weights="ResNet50_Weights.DEFAULT")

        # Get the number of features in the original fc layer
        in_features = self.backbone.fc.in_features

        # Remove the original fully connected layer and add a custom classifier
        self.backbone.fc = nn.Identity()  # Remove the default classification head
        self.fc = nn.Linear(in_features, num_classes)  # Custom classifier

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Hyperparameters
        self.save_hyperparameters({"num_classes": num_classes, "lr": lr})
        self.lr = lr

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Pass through the backbone
        x = self.backbone(x)  # Output shape: (batch_size, 2048)

        # Pass directly through the custom classification layer
        x = self.fc(x)  # Output shape: (batch_size, num_classes)
        return x

    def training_step(self, batch, batch_idx):
        data, target, _ = batch  # Adjust if your dataset returns additional items
        preds = self(data)
        loss = self.criterion(preds, target)

        acc = self.train_accuracy(preds, target)
        self.log("train_loss", loss, on_epoch=True, batch_size=data.size(0))
        self.log("train_acc", acc, on_epoch=True, batch_size=data.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch

        with torch.no_grad():
            preds = self(data)
        loss = self.criterium(preds, target)

        acc = self.val_accuracy(preds, target)
        self.log("val_loss", loss, on_epoch=True, batch_size=data.size(0))
        self.log("val_acc", acc, on_epoch=True, batch_size=data.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    num_classes = cfg.model.num_classes
    lr = cfg.train.lr

    # Instantiate the model
    model = ResNetModel(num_classes, lr)

    # Save the model as ONNX
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor

    model.to_onnx(
        file_path="resnet18.onnx",
        input_sample=dummy_input,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_params=True,  # Ensure weights are exported
    )

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
