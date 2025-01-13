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

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def train(cfg: DictConfig):
    """
    Train a ResNetModel using PyTorch Lightning.

    Args:
        cfg (DictConfig): Configuration loaded from Hydra.
    """
    # Initialize wandb
    wandb.init(
        project="corrupt_mnist",
        config={"lr": cfg.train.lr, "batch_size": cfg.train.batch_size, "epochs": cfg.train.epochs},
    )
    wandb_logger = WandbLogger(
        project="pokemon_classifier",
        name="add this to the config",  # Optional: Name the specific run
    )
    wandb_logger.experiment.config.update({
        "lr": cfg.train.lr, 
        "batch_size": cfg.train.batch_size, 
        "epochs": cfg.train.epochs
    })

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

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min", filename="m{cfg.train.epochs:02d}-{val_loss:.2f}"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    
    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.train.devices,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()


#TODO
# Add logging to wandb (maybe done - need testing)
# save of model during training (maybe done)
# add early stopping (maybe done)
# add scheduler?
# add logging instead of printing
# dvc

# remove warnings
# optimize/profiling