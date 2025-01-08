import torch
from torch import nn
from torch import optim
from data import Dataset
from model import ResNetModel
import hydra
from omegaconf import DictConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate(model_checkpoint: str, batch_size: int) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(f"Loading model from checkpoint: {model_checkpoint}")

    # Load the trained model
    model = ResNetModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()  # Set the model to evaluation mode

    # Load test dataset
    test_set = Dataset(mode="test")
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    # Evaluation logic
    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient calculations for evaluation
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

    print(f"Test accuracy: {correct / total:.2%}")

@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    model_checkpoint = cfg.evaluation.model_checkpoint
    batch_size = cfg.dataset.batch_size  # Fetch batch size from config
    evaluate(model_checkpoint, batch_size)

if __name__ == "__main__":
    main()