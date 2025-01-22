import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from data import PokeDataset
from model import ResNetModel

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BUCKET_NAME = "mlops_bucket100"

def evaluate(model, batch_size: int) -> None:
    """Evaluate a trained model on the test dataset."""
    print("Starting evaluation...")
    # Load the model weights
    model.eval()

    # Prepare the dataset and dataloader
    test_set = PokeDataset(bucket_name=BUCKET_NAME, mode="test", transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    correct, total = 0, 0
    with torch.no_grad():
        for images, targets, _ in test_dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            print(f"Batch accuracy: {correct / total:.2%}")

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.2%}")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    processor = AutoImageProcessor.from_pretrained("imjeffhi/pokemon_classifier")
    model = AutoModelForImageClassification.from_pretrained("imjeffhi/pokemon_classifier")
    model.to(DEVICE)

    evaluate(model, cfg.batch_size)
