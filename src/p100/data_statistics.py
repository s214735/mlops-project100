from collections import Counter
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from data import PokeDataset 
import torch 
import random


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

    # Generate plots for class distribution
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

    # Generate and save a grid of sample images with labels
    generate_image_grid_with_random_classes(train_dataset, "combined_train_images.png", num_images=10)
    print("Combined image grid saved as 'combined_train_images.png'")

def generate_image_grid_with_random_classes(dataset, output_file, num_images=10):
    """Generate a grid of images from random classes and save as a single image."""
    cols = 5
    rows = (num_images + cols - 1) // cols  # Determine number of rows
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

    # Shuffle dataset indices for randomness
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Dictionary to track selected images for unique classes
    selected_classes = {}

    # Iterate through shuffled dataset
    for idx in indices:
        if len(selected_classes) >= num_images:
            break  # Stop once we have enough images

        img, label, class_name = dataset[idx][:3]  # Unpack image, label, and class_name

        # Add one image per class
        if class_name not in selected_classes:
            selected_classes[class_name] = img

    # Plot the selected images
    for i, (class_name, img) in enumerate(selected_classes.items()):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)  # Convert tensor to PIL image if necessary
        
        axes[i].imshow(img)
        axes[i].set_title(f"{class_name}", fontsize=10)  # Use class_name as the title
        axes[i].axis("off")  # Remove axes for a cleaner look

    # Hide unused axes
    for idx in range(len(selected_classes), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    dataset_statistics()