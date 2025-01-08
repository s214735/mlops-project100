import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from model import ResNetModel
from data import PokeDataset

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def train(cfg: DictConfig):
    """
    Train a ResNetModel using PyTorch Lightning.

    Args:
        cfg (DictConfig): Configuration loaded from Hydra.
    """
    # Initialize dataset and dataloaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = PokeDataset(
        processed_data_path=Path(cfg.data.processed_path),
        mode="train",
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers
    )

    val_dataset = PokeDataset(
        processed_data_path=Path(cfg.data.processed_path),
        mode="val",
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers
    )

    # Initialize model
    model = ResNetModel(num_classes=cfg.model.num_classes, lr=cfg.train.lr)

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.train.devices,
        log_every_n_steps=cfg.train.log_every_n_steps
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()


#TODO
# Add logging to wandb
# remove warnings
# optimize
# 