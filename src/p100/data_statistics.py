import os
from collections import Counter
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from data import PokeDataset  # Replace with your dataset import
import numpy as np


def update_markdown_file(markdown_path, replacements):
    """
    Update the markdown file by replacing placeholders with actual values.
    Args:
        markdown_path (str): Path to the markdown file.
        replacements (dict): Dictionary of placeholder keys and their replacement values.
    """
    with open(markdown_path, "r") as file:
        content = file.read()

    # Replace each placeholder with its corresponding value
    for key, value in replacements.items():
        content = content.replace(key, str(value))

    with open(markdown_path, "w") as file:
        file.write(content)


def class_distribution_plot(train_dataset, val_dataset, test_dataset, output_file):
    """Generate class distribution plot."""
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
    val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    val_counts = Counter(val_labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    axes[0].bar(train_counts.keys(), train_counts.values(), color="blue")
    axes[0].set_title("Train Dataset Class Distribution")
    axes[1].bar(val_counts.keys(), val_counts.values(), color="green")
    axes[1].set_title("Validation Dataset Class Distribution")
    axes[2].bar(test_counts.keys(), test_counts.values(), color="red")
    axes[2].set_title("Test Dataset Class Distribution")
    plt.tight_layout()
    fig.savefig(output_file)
    plt.close()


def generate_image_grid_with_random_classes(dataset, output_file, num_images=10):
    """Generate a grid of images from random classes and save as a single image."""
    import random
    cols = 5
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axes = axes.flatten()
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_classes = {}
    for idx in indices:
        if len(selected_classes) >= num_images:
            break
        img, label, class_name = dataset[idx][:3]
        if class_name not in selected_classes:
            selected_classes[class_name] = img
    for i, (class_name, img) in enumerate(selected_classes.items()):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        axes[i].imshow(img)
        axes[i].set_title(f"{class_name}", fontsize=10)
        axes[i].axis("off")
    for idx in range(len(selected_classes), len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def dataset_statistics(datadir: str = "data/processed", markdown_path: str = "src/p100/data_statistics.md") -> None:
    """Compute dataset statistics and update the markdown file."""
    train_dataset = PokeDataset(datadir, mode="train", transform=transforms.ToTensor())
    test_dataset = PokeDataset(datadir, mode="test", transform=transforms.ToTensor())
    val_dataset = PokeDataset(datadir, mode="val", transform=transforms.ToTensor())

    # Compute statistics
    train_count = len(train_dataset)
    train_shape = train_dataset[0][0].shape
    val_count = len(val_dataset)
    val_shape = val_dataset[0][0].shape
    test_count = len(test_dataset)
    test_shape = test_dataset[0][0].shape

    # Print dataset info
    print(f"-----Train dataset-----")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print(f"Number of classes: {len(np.unique(train_dataset.targets))}")
    print("\n")
    print(f"-----Test dataset-----")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")
    print(f"Number of classes: {len(np.unique(test_dataset.targets))}")
    print("\n")
    print(f"-----Val dataset-----")
    print(f"Number of images: {len(val_dataset)}")
    print(f"Image shape: {val_dataset[0][0].shape}")
    print(f"Number of classes: {len(np.unique(val_dataset.targets))}")

    # Generate plots
    class_distribution_plot(train_dataset, val_dataset, test_dataset, "class_distribution.png")
    generate_image_grid_with_random_classes(train_dataset, "combined_train_images.png", num_images=10)

    # Update placeholders in markdown
    replacements = {
        "[TRAIN_COUNT]": train_count,
        "[TRAIN_SHAPE]": train_shape,
        "[VAL_COUNT]": val_count,
        "[VAL_SHAPE]": val_shape,
        "[TEST_COUNT]": test_count,
        "[TEST_SHAPE]": test_shape,
    }
    update_markdown_file(markdown_path, replacements)

    print(f"Markdown report updated at {markdown_path}")


if __name__ == "__main__":
    dataset_statistics()
