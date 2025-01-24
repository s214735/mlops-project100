import wandb
from p100.utils import get_wandb_api_key
from p100.evaluate import get_latest_model_path, load_model

import numpy as np
import torch

from p100.model import ResNetModel
from p100.data import PokeDataset
from torchvision import transforms
from torch.utils.data import DataLoader


import warnings
import umap
import matplotlib.pyplot as plt

BUCKET_NAME = "mlops_bucket100"
PREFIX = "data/processed/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_PATH = "p100-org/wandb-registry-Pokemon"
MODEL_ALIAS = "Model:latest"

warnings.filterwarnings("ignore")


def main():
    wandb.login(key=get_wandb_api_key())

    model = ResNetModel(num_classes=18, lr=1).to(device)
    model_checkpoint = get_latest_model_path(PROJECT_PATH, MODEL_ALIAS)

    # Load the model weights
    model = load_model(model, model_checkpoint)
    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = PokeDataset(bucket_name=BUCKET_NAME, mode="test", transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=32)

    features = []
    labels = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar(scatter, ticks=range(int(labels.min()), int(labels.max()) + 1))
    plt.title('UMAP projection of the test dataset')
    plt.savefig('mlops-project100-1/images/umap_projection.png')

if __name__ == "__main__":
    main()
