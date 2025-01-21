from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import io
import numpy as np
from google.cloud import storage

BUCKET_NAME = "mlops_bucket100"

class PokeDataset(Dataset):
    def __init__(self, bucket_name: str, processed_data_path: str = "data/processed", mode: str = "train", transform=None):
        self.bucket_name = bucket_name
        self.data_path = processed_data_path
        self.mode = mode
        self.transform = transform or transforms.ToTensor()

        self.data = []  # Stores image paths in the bucket
        self.targets = []  # Stores class indices
        self.class_names = []  # Stores class names

        self._load_dataset()

        # GCS client is initialized as None to avoid pickling issues
        self.client = None
        self.bucket = None

    def _initialize_gcs_client(self):
        """Initialize the GCS client lazily."""
        if self.client is None or self.bucket is None:
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)

    def _load_dataset(self):
        """Load the dataset structure from the GCS bucket."""
        # Create a temporary client to load dataset metadata during initialization
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        prefix = f"{self.data_path}/{self.mode}/"
        blobs = bucket.list_blobs(prefix=prefix)

        # Use a dictionary to track class indices
        class_to_index = {}
        for blob in blobs:
            if blob.name.endswith('/'):
                continue

            # Parse the class name from the file path
            parts = blob.name.split('/')
            class_name = parts[-2]

            if class_name not in class_to_index:
                class_to_index[class_name] = len(class_to_index)

            self.data.append(blob.name)
            self.targets.append(class_to_index[class_name])
            self.class_names.append(class_name)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        self._initialize_gcs_client()  # Ensure client is initialized in the worker

        blob_name = self.data[idx]
        target = self.targets[idx]
        class_name = self.class_names[idx]

        blob = self.bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

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