import json

import hydra
import pytorch_lightning as pl
import torch
from google.cloud import secretmanager
from model import ResNetModel
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
from data import PokeDataset

BUCKET_NAME = "mlops_bucket100"


def get_wandb_api_key() -> str:
    """
    Retrieve the W&B API key from Google Secret Manager.
    """
    client = secretmanager.SecretManagerServiceClient()

    # Construct the secret name
    secret_name = "wandb-api-key"
    project_id = "level-oxygen-447714-d3"
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

    # Access the secret value
    response = client.access_secret_version(name=secret_path)
    secret_payload = response.payload.data.decode("UTF-8")

    # Parse the JSON and extract the key
    secret_dict = json.loads(secret_payload)
    api_key = secret_dict.get("wandb-api-key")

    return api_key


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def train(cfg: DictConfig):
    """
    Train a ResNetModel using PyTorch Lightning.

    Args:
        cfg (DictConfig): Configuration loaded from Hydra.
    """
    my_key = get_wandb_api_key()
    wandb.login(key=my_key)
    # Initialize wandb
    wandb.finish()
    wandb_logger = WandbLogger(
        project="pokemon_classifier",
    )
    wandb_logger.experiment.config.update(
        {
            "lr": cfg.train.lr,
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "model": "ResNet50",
        }
    )

    # Initialize dataset and dataloaders
    transform_train = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transforms_test = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = PokeDataset(BUCKET_NAME, mode="test", transform=transform_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        persistent_workers=True,
    )

    val_dataset = PokeDataset(BUCKET_NAME, mode="val", transform=transforms_test)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        persistent_workers=True,
    )

    # Initialize model
    model = ResNetModel(num_classes=cfg.model.num_classes, lr=cfg.train.lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min", filename="m{val_loss:.2f}"
    )

    # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.train.devices,
        log_every_n_steps=cfg.train.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],  # , early_stopping_callback],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Create and log artifact
    artifact = wandb.Artifact("pokemon_classifier_model", type="model")
    artifact.add_dir("./models")
    wandb_logger.experiment.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    train()
