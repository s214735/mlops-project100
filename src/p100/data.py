from pathlib import Path
import os
import typer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class Dataset(Dataset):
    """My custom dataset."""

    def __init__(self, processed_data_path: Path, mode: str, transform = None) -> None:
        # Define the path to the raw data
        self.data_path = processed_data_path
        # Define lists to store the data and targets
        self.data = []
        self.targets = []
        self.transform = transform        
        # Define the mode of the dataset
        self.mode = mode
        # Define the path to the split data
        self.split_dir = os.path.join(self.data_path, self.mode)

        # Traverse the folder structure
        for class_name in os.listdir(self.split_dir):
            class_path = os.path.join(self.split_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.targets.append(class_name)
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the data and target at the given index."""
        img_path = self.data[idx]
        target = self.targets[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # You can implement specific preprocessing logic here, such as saving processed data
        print(f"Preprocessing done. Processed data saved to {output_folder}")


if __name__ == "__main__":
    root_dir = r"C:\Users\emilg\Desktop\DTU\1. Semester\Machine Learning Operations\Project100\MLOps-project100\data\processed"
    dataset = Dataset(root_dir,"train",transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data, target in dataloader:
        print(data.shape, target)
        break



