import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

from data import PokeDataset
from model import ResNetModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PROJECT_PATH = "p100-org/wandb-registry-Pokemon"  # Specify your W&B project path

def get_latest_model_path(project_path: str) -> str:
    """Retrieve the path to the latest model from W&B."""
    api = wandb.Api()
    model_registry = api.model(f"{project_path}/Model")
    latest_version = model_registry.version("latest")
    model_artifact = latest_version.use()
    downloaded_path = model_artifact.download()
    return downloaded_path

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
    model_checkpoint = get_latest_model_path(PROJECT_PATH)
    batch_size = cfg.evaluate.batch_size
    evaluate(model_checkpoint, batch_size)

if __name__ == "__main__":
    main()
