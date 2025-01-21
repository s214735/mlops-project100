from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import io
import numpy as np
from google.cloud import storage

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import io
import numpy as np
from google.cloud import storage

BUCKET_NAME = "mlops_bucket100"
CACHE_DIR = "data_cache"  # Directory to cache images locally


class PokeDataset(Dataset):
    """Custom dataset to load data from a Google Cloud Storage bucket."""

    def __init__(self, bucket_name: str, processed_data_path: str = "data/processed", mode: str = "train", transform=None):
        """
        :param bucket_name: Name of the GCS bucket.
        :param processed_data_path: Path to the processed data within the bucket.
        :param mode: Dataset mode (e.g., 'train', 'test').
        :param transform: Transformations to apply to the images.
        """
        self.bucket_name = bucket_name
        self.data_path = processed_data_path
        self.mode = mode
        self.transform = transform or transforms.ToTensor()

        self.data = []  # Stores local file paths
        self.targets = []  # Stores class indices
        self.class_names = []  # Stores class names

        # Use a dictionary to track class indices
        self.class_to_index = {}

        # Cache dataset locally
        self._cache_dataset()

    def _initialize_client(self):
        """Initialize GCS client."""
        if not hasattr(self, '_client'):
            self._client = storage.Client()  # Lazy initialization
        return self._client

    def _cache_dataset(self):
        """Download and cache dataset locally."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        client = self._initialize_client()
        bucket = client.bucket(self.bucket_name)
        prefix = f"{self.data_path}/{self.mode}/"
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if blob.name.endswith('/'):
                continue

            # Parse the class name from the file path
            parts = blob.name.split('/')
            class_name = parts[-2]

            # Assign a class index if it's new
            if class_name not in self.class_to_index:
                self.class_to_index[class_name] = len(self.class_to_index)

            # Define local file path
            local_path = os.path.join(CACHE_DIR, os.path.basename(blob.name))
            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)  # Download if not cached

            # Append to dataset lists
            self.data.append(local_path)  # Use local path
            self.targets.append(self.class_to_index[class_name])
            self.class_names.append(class_name)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Fetch the image, target, and class name by index."""
        image_path = self.data[idx]
        target = self.targets[idx]
        class_name = self.class_names[idx]

        try:
            # Open the image
            image = Image.open(image_path).convert('RGB')
        except (IOError, Image.UnidentifiedImageError):
            print(f"Skipping corrupted image: {image_path}")
            return self.__getitem__((idx + 1) % len(self))  # Load the next image instead

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, target, class_name




if __name__ == "__main__":
    train_dataset = PokeDataset(BUCKET_NAME, mode="train", transform=transforms.ToTensor())
    test_dataset = PokeDataset(BUCKET_NAME, mode="test", transform=transforms.ToTensor())
    val_dataset = PokeDataset(BUCKET_NAME, mode="val", transform=transforms.ToTensor())

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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
    print(f"Min label: {min(train_dataset.targets)}. Max label: {max(train_dataset.targets)}")
    print("\n")
    print(f"-----Test dataset-----")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")
    print(f"Number of classes: {len(np.unique(test_dataset.targets))}")
    print(f"Min label: {min(test_dataset.targets)}. Max label: {max(test_dataset.targets)}")
    print("\n")
    print(f"-----Val dataset-----")
    print(f"Number of images: {len(val_dataset)}")
    print(f"Image shape: {val_dataset[0][0].shape}")
    print(f"Number of classes: {len(np.unique(val_dataset.targets))}")
    print(f"Min label: {min(val_dataset.targets)}. Max label: {max(val_dataset.targets)}")
    
    for data, target, class_name in dataloader:
        print(data.shape, target, class_name)
        break