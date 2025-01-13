import os
import sys

import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))
from p100.model import ResNetModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def test_model():
    path = "models/testmodel.pth"
    model = ResNetModel().to(DEVICE)
    torch.save(model.state_dict(), path)
    print("Saved dummy model")

def test_model2():
    model = ResNetModel()
    x = torch.randn(1,3,128, 128)
    y = model(x)
    assert y.shape == (1,1000)

if __name__ == "__main__":
    test_model()
