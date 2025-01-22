import os
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import warnings

from data import PokeDataset
from model import ResNetModel
from p100.utils import get_wandb_api_key

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PROJECT_PATH = "p100-org/wandb-registry-Pokemon"  # Specify your W&B project path
MODEL_ALIAS = "Model:latest"  # Use the alias for the latest model version

def get_file_with_lowest_val_loss(artifact_path: str) -> str:
    """Find the model file with the lowest val_loss based on the filename."""
    lowest_loss = float('inf')
    best_model_path = None

    for file_name in os.listdir(artifact_path):
        if file_name.startswith("mval_loss=") and file_name.endswith(".ckpt"):
            # Extract the val_loss from the filename
            try:
                val_loss = float(file_name.split("=")[1].replace(".ckpt", ""))
                if val_loss < lowest_loss:
                    lowest_loss = val_loss
                    best_model_path = os.path.join(artifact_path, file_name)
            except ValueError:
                continue  # Skip files that don't match the format
    
    if best_model_path is None:
        raise FileNotFoundError("No valid model file with val_loss found.")
    
    print(f"Selected model: {best_model_path} with val_loss: {lowest_loss}")
    return best_model_path

def get_latest_model_path(project_path: str, alias: str) -> str:
    """Retrieve the path to the latest model from W&B."""
    my_key = get_wandb_api_key()
    wandb.login(key=my_key)

    api = wandb.Api()
    model_artifact = api.artifact(f"{project_path}/{alias}")
    downloaded_path = model_artifact.download()
    print(f"Model artifact downloaded to: {downloaded_path}")

    # Locate the file with the lowest val_loss
    return get_file_with_lowest_val_loss(downloaded_path)

def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """Load model weights from a file."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def evaluate(model_checkpoint: str, batch_size: int, device=DEVICE) -> None:
    """Evaluate a trained model on the test dataset."""
    print("Starting evaluation...")

    # Initialize the model
    model = ResNetModel(num_classes=1000, lr=1).to(device)

    # Load the model weights
    model = load_model(model, model_checkpoint)
    model.eval()

    # Prepare the dataset and dataloader
    test_set = PokeDataset(mode="test", transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    correct, total = 0, 0
    with torch.no_grad():
        for images, targets, _ in test_dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.2%}")


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    # Fetch the latest model path
    model_checkpoint = get_latest_model_path(PROJECT_PATH, MODEL_ALIAS)
    batch_size = cfg.evaluate.batch_size
    evaluate(model_checkpoint, batch_size)

if __name__ == "__main__":
    main()
