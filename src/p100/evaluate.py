import torch
import typer
from torch import nn
from torch import optim
from data import corrupt_mnist
from model import ResNetModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(f"Loading model from checkpoint: {model_checkpoint}")

    # Load the trained model
    model = ResNetModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))
    model.eval()  # Set the model to evaluation mode

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    # Disable gradient calculations for evaluation  
    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)