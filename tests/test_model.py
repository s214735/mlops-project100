import torch
from p100.model import ResNetModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def test_model():
    model = ResNetModel()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 18)  # Check if the output shape is as expected (batch size 1, 1000 classes)


if __name__ == "__main__":
    test_model()
