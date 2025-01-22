import os
import hydra
import torch
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

def get_latest_model_path(project_path: str, alias: str) -> str:
    """Retrieve the path to the latest model from W&B."""
    my_key = get_wandb_api_key()
    wandb.login(key=my_key)

    api = wandb.Api()
    model_artifact = api.artifact(f"{project_path}/{alias}")
    downloaded_path = model_artifact.download()
    print(f"Model artifact downloaded to: {downloaded_path}")

    # Locate the specific model file in the artifact directory
    for file_name in os.listdir(downloaded_path):
        if file_name.endswith((".pth", ".ckpt")):  # Adjust extensions as needed
            return os.path.join(downloaded_path, file_name)
    
    raise FileNotFoundError("No model file found in the artifact directory.")

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
    model = ResNetModel(num_classes=18, lr=1).to(device)

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
