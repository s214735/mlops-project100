from pathlib import Path
import os
#import typer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import typer
from collections import Counter


class PokeDataset(Dataset):
    """My custom dataset."""

    def __init__(self, processed_data_path: Path = "data/processed", mode: str = "train", transform = transforms.ToTensor()) -> None:
        # Define the path to the raw data

        self.data_path = processed_data_path
        # Define lists to store the data and targets
        self.data = []
        self.targets = []

        self.class_names = []  # To store class names for each sample
        self.transform = transform  # Transformation function (e.g., normalization, augmentation)
        # Define the mode (train, test, etc.)

        self.mode = mode
        # Define the path to the split data
        self.split_dir = os.path.join(self.data_path, self.mode)

        # Traverse the folder structure
        for index, class_name in enumerate(os.listdir(self.split_dir)):
            class_path = os.path.join(self.split_dir, class_name)

            if os.path.isdir(class_path):  # Check if it is a directory (class folder)
                for img_name in os.listdir(class_path):  # Iterate through images in the class folder
                    img_path = os.path.join(class_path, img_name)  # Get the image path
                    self.data.append(img_path)  # Add image path to data list
                    self.targets.append(index)  # Add class index to targets list
                    self.class_names.append(class_name)  # Add class name to class_names list

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the data and target at the given index."""
        img_path = self.data[idx]
        target = self.targets[idx]
        class_name = self.class_names[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target, class_name

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # You can implement specific preprocessing logic here, such as saving processed data
        print(f"Preprocessing done. Processed data saved to {output_folder}")


def dataset_statistics(datadir: str = "data/processed") -> None:
    """Compute dataset statistics and save class distribution plots."""
    train_dataset = PokeDataset(datadir, mode="train", transform=transforms.ToTensor())
    test_dataset = PokeDataset(datadir, mode="test", transform=transforms.ToTensor())
    val_dataset = PokeDataset(datadir, mode="val", transform=transforms.ToTensor())

    # Print dataset info
    print(f"-----Train dataset-----")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print(f"-----Test dataset-----")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")
    print("\n")
    print(f"-----Val dataset-----")
    print(f"Number of images: {len(val_dataset)}")
    print(f"Image shape: {val_dataset[0][0].shape}")

    # Count class occurrences in each dataset
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
    val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    val_counts = Counter(val_labels)

    # Generate plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    axes[0].bar(train_counts.keys(), train_counts.values(), color='blue')
    axes[0].set_title('Train Dataset Class Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Frequency')

    axes[1].bar(val_counts.keys(), val_counts.values(), color='green')
    axes[1].set_title('Validation Dataset Class Distribution')
    axes[1].set_xlabel('Class')

    axes[2].bar(test_counts.keys(), test_counts.values(), color='red')
    axes[2].set_title('Test Dataset Class Distribution')
    axes[2].set_xlabel('Class')

    plt.tight_layout()

    # Save the plots
    fig.savefig("class_distribution.png")
    print("Class distribution plot saved as 'class_distribution.png'")


if __name__ == "__main__":
    typer.run(dataset_statistics)


if __name__ == "__main__":
    dataset = Dataset(transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data, target, class_name in dataloader:
        print(data.shape, target, class_name)
        break
    
