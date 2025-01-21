from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import io
import numpy as np
from google.cloud import storage

BUCKET_NAME = "mlops_bucket100"

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

        self.data = []  # Stores image paths in the bucket
        self.targets = []  # Stores class indices
        self.class_names = []  # Stores class names

        # Use a dictionary to track class indices
        self.class_to_index = {}

        # Load dataset file paths and targets
        self._load_dataset()

    def _initialize_client(self):
        """Initialize GCS client and bucket."""
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        return bucket

    def _load_dataset(self):
        """Load the dataset structure from the GCS bucket."""
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        prefix = f"{self.data_path}/{self.mode}/"
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            # Skip directories (GCS directories are implied by paths ending in '/')
            if blob.name.endswith('/'):
                continue

            # Parse the class name from the file path
            parts = blob.name.split('/')
            class_name = parts[-2]  # Assume class name is the second-to-last folder

            # Assign a class index if it's new
            if class_name not in self.class_to_index:
                self.class_to_index[class_name] = len(self.class_to_index)

            # Append to dataset lists
            self.data.append(blob.name)  # Full GCS path
            self.targets.append(self.class_to_index[class_name])
            self.class_names.append(class_name)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Fetch the image and target by index."""
        # Lazily initialize the GCS bucket
        bucket = self._initialize_client()

        # Get the blob name and target
        blob_name = self.data[idx]
        target = self.targets[idx]
        class_name = self.class_names[idx]

        # Fetch the image blob from GCS
        blob = bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()

        # Open the image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

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