import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch
from torch import nn
from torchvision.models import resnet50
from model import ResNetModel
from data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl


def collate_fn(batch):
    """Custom collate function to return only the first two outputs (data, target)."""
    data, target, _ = zip(*batch)  # Unzip the batch and ignore class_name
    return torch.stack(data), torch.tensor(target)  # Stack data and convert target to tensor

class CustomDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for loading the custom dataset."""

    def __init__(self, data_path: Path, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize if necessary
            transforms.ToTensor(),
        ])
        self.train_dataset = Dataset(self.data_path, mode="train", transform=transform)
        self.val_dataset = Dataset(self.data_path, mode="val", transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn  # Use custom collate_fn to only return (data, target)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn  # Use custom collate_fn here too
        )




@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    # Initialize the DataModule
    data_module = CustomDataModule(
        data_path=Path("data/processed"), 
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
