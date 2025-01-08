from pathlib import Path  # Import Path for easier path handling
import os  # Import os for file system operations
#import typer  # Commented out, might be used for command line interface
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader for handling data in PyTorch
from PIL import Image  # Import Image from PIL for image loading and processing
from torchvision import transforms  # Import transforms for data augmentation and normalization

class Dataset(Dataset):  # Custom dataset class inheriting from PyTorch Dataset class
    ###Custom dataset###

    def __init__(self, processed_data_path: Path = r"data\processed", mode: str = "train", transform = transforms.ToTensor()) -> None:
        # Define the path to the processed data
        self.data_path = processed_data_path
        # Define lists to store image paths (data) and corresponding labels (targets)
        self.data = []
        self.targets = []
        self.class_names = []  # To store class names for each sample
        self.transform = transform  # Transformation function (e.g., normalization, augmentation)
        # Define the mode (train, test, etc.)
        self.mode = mode
        # Define the directory for the current dataset split (e.g., train, val)
        self.split_dir = os.path.join(self.data_path, self.mode)

        # Traverse the folder structure where images are stored
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
        return len(self.data)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        """Return the data and target at the given index."""
        img_path = self.data[idx]  # Get the image path at the specified index
        target = self.targets[idx]  # Get the target label at the specified index
        class_name = self.class_names[idx]  # Get the class name at the specified index
        image = Image.open(img_path).convert('RGB')  # Open and convert image to RGB mode

        if self.transform:
            image = self.transform(image)  # Apply transformation (e.g., convert to tensor)

        return image, target, class_name  # Return the processed image, target, and class name

# Main entry point of the script
if __name__ == "__main__":
    # Instantiate the custom dataset with a tensor transformation
    dataset = Dataset(transform=transforms.ToTensor())
    # Create a DataLoader to load the dataset in batches
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Iterate through the DataLoader
    for data, target, class_name in dataloader:
        print(data.shape, target, class_name)  # Print the shape of the data, target labels, and class names
        break  # Break after the first batch for demonstration



