from pathlib import Path
import os
#import typer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

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


if __name__ == "__main__":
    dataset = Dataset(transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data, target, class_name in dataloader:
        print(data.shape, target, class_name)
        break



