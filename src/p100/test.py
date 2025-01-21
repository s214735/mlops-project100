import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import PokeDataset

BUCKET_NAME = "mlops_bucket100"


def main():
    transform_train = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets and dataloaders
    train_dataset = PokeDataset(BUCKET_NAME, mode="train", transform=transform_train)

    if torch.cuda.is_available():
        num_workers = 6
    else:
        num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    device = torch.device("cuda")

    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        print(batch_idx)


if __name__ == "__main__":
    main()
