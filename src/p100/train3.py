import json

# Ignore warnings
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from data3 import PokeDataset
from google.cloud import secretmanager
from torch.utils.data import DataLoader
from torchvision import models, transforms

import wandb

warnings.filterwarnings("ignore")

BUCKET_NAME = "mlops_bucket100"


# Function to get W&B API key from Google Secret Manager
def get_wandb_api_key() -> str:
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "wandb-api-key"
    project_id = "level-oxygen-447714-d3"  # Replace with your GCP project ID
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=secret_path)
    secret_payload = response.payload.data.decode("UTF-8")
    secret_dict = json.loads(secret_payload)
    return secret_dict.get("wandb-api-key")


# Define the ResNet model class
class ResNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNetModel, self).__init__()
        self.backbone = models.resnet50(weights="ResNet50_Weights.DEFAULT" if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Training function
def train_one_epoch(model, dataloader, optimizer, criterion, epoch, log_every):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    for batch_idx, (data, target, _) in enumerate(dataloader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(data)
            loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % log_every == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, Correct: {correct}, Total: {total}")

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    return epoch_loss, epoch_accuracy


# Validation function
def validate(model, dataloader, criterion, epoch, log_every):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if batch_idx % log_every == 0:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, "
                    f"Validation Loss: {loss.item()}, Correct: {correct}, Total: {total}"
                )

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    return epoch_loss, epoch_accuracy


# Main training script
def main():
    # Initialize W&B
    wandb_api_key = get_wandb_api_key()
    wandb.login(key=wandb_api_key)
    wandb.init(project="pokemon_classifier")

    # Configuration
    cfg = {
        "lr": 0.00001,
        "batch_size": 16,
        "epochs": 10,
        "num_classes": 1000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "weight_decay": 1e-4,
    }
    wandb.config.update(cfg)

    # Data transforms
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

    transform_test = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets and dataloaders
    train_dataset = PokeDataset(BUCKET_NAME, mode="train", transform=transform_train)
    val_dataset = PokeDataset(BUCKET_NAME, mode="val", transform=transform_test)

    if torch.cuda.is_available():
        num_workers = 6
    else:
        num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Model, criterion, optimizer
    model = ResNetModel(num_classes=cfg["num_classes"]).to(cfg["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )  # Add weight decay to the optimizer

    # Training loop
    print("Starting training...")
    for epoch in range(cfg["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, log_every=10)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, log_every=10)

        print(
            f"Epoch {epoch+1}/{cfg['epochs']}\n"
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%\n"
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%"
        )

        # Log metrics to W&B
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    # Save model
    torch.save(model.state_dict(), "resnet18_pokemon.pth")
    wandb.save("resnet18_pokemon.pth")
    wandb.finish()


if __name__ == "__main__":
    main()
