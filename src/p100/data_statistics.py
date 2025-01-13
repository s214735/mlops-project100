from collections import Counter
from torchvision import transforms

import matplotlib.pyplot as plt
from data import PokeDataset  

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
    dataset_statistics()