import torch
from torch.utils.data import DataLoader
from data import Dataset  # Custom dataset implementation
from model import ResNetModel  # Custom ResNet model implementation
import hydra
from omegaconf import DictConfig
from torchvision import transforms

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def evaluate(model_checkpoint: str, batch_size: int) -> None:
    """Evaluate a trained model on the test dataset."""
    print("Starting evaluation...")
    print(f"Loading model from checkpoint: {model_checkpoint}")

    model = ResNetModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))  # Load model weights
    model.eval()

    test_set = Dataset(mode="test", transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    correct, total = 0, 0
    with torch.no_grad():
        for images, targets, _ in test_dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.2%}")

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    model_checkpoint = cfg.evaluation.model_checkpoint
    batch_size = cfg.dataset.batch_size
    evaluate(model_checkpoint, batch_size)

if __name__ == "__main__":
    main()