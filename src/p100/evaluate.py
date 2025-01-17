import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import PokeDataset
from .model import ResNetModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load model weights from a .ckpt or .pth file."""
    print(f"Loading model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

    if "state_dict" in checkpoint:  # Case for .ckpt files
        model.load_state_dict(checkpoint["state_dict"])
    else:  # Case for .pth files
        model.load_state_dict(checkpoint)

    return model


def evaluate(model_checkpoint: str, batch_size: int) -> None:
    """Evaluate a trained model on the test dataset."""
    print("Starting evaluation...")

    # Initialize the model
    model = ResNetModel(num_classes=1000, lr=1).to(DEVICE)

    # Load the model weights
    model = load_model(model, model_checkpoint)
    model.eval()

    # Prepare the dataset and dataloader
    test_set = PokeDataset(mode="test", transform=transforms.ToTensor())
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
    model_checkpoint = cfg.evaluate.model_checkpoint
    batch_size = cfg.evaluate.batch_size
    evaluate(model_checkpoint, batch_size)


if __name__ == "__main__":
    main()
