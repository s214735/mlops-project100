import io

from google.cloud import storage
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BUCKET_NAME = "mlops_bucket100"
PROJECT_NAME = "level-oxygen-447714-d3"


class PokeDataset(Dataset):
    """Custom dataset to load data from a Google Cloud Storage bucket."""

    def __init__(
        self, bucket_name: str, processed_data_path: str = "data/processed", mode: str = "train", transform=None
    ):
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

        # Client and bucket will be initialized lazily in each worker
        self.client = None
        self.bucket = None

        # Load dataset file paths and targets
        self._load_dataset()

    def _get_client_and_bucket(self):
        """Initialize the GCS client and bucket lazily."""
        if self.client is None or self.bucket is None:
            self.client = storage.Client(project=PROJECT_NAME)
            self.bucket = self.client.bucket(self.bucket_name)

    def _load_dataset(self):
        """Load the dataset structure from the GCS bucket."""
        # Use a temporary client to load file paths
        temp_client = storage.Client()
        temp_bucket = temp_client.bucket(self.bucket_name)

        prefix = f"{self.data_path}/{self.mode}/"
        blobs = temp_bucket.list_blobs(prefix=prefix)

        # Use a dictionary to track class indices
        class_to_index = {}
        for blob in blobs:
            # Skip directories (GCS directories are implied by paths ending in '/')
            if blob.name.endswith("/"):
                continue

            # Parse the class name from the file path
            parts = blob.name.split("/")
            class_name = parts[-2]  # Assume class name is the second-to-last folder

            # Assign a class index if it's new
            if class_name not in class_to_index:
                class_to_index[class_name] = len(class_to_index)

            # Append to dataset lists
            self.data.append(blob.name)  # Full GCS path
            self.targets.append(class_to_index[class_name])
            self.class_names.append(class_name)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Fetch the image and target by index."""
        # Initialize the client and bucket for this worker if necessary
        self._get_client_and_bucket()

        # Get the blob name and target
        blob_name = self.data[idx]
        target = self.targets[idx]
        class_name = self.class_names[idx]

        # Fetch the image blob from GCS
        blob = self.bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()

        # Open the image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, target, class_name

    def __getstate__(self):
        """Exclude unpicklable attributes from the state."""
        state = self.__dict__.copy()
        state["client"] = None
        state["bucket"] = None
        return state

    def __setstate__(self, state):
        """Restore the state and reset unpicklable attributes."""
        self.__dict__.update(state)
        self.client = None
        self.bucket = None


if __name__ == "__main__":
    train_dataset = PokeDataset(BUCKET_NAME, mode="train", transform=transforms.ToTensor())
    test_dataset = PokeDataset(BUCKET_NAME, mode="test", transform=transforms.ToTensor())
    val_dataset = PokeDataset(BUCKET_NAME, mode="val", transform=transforms.ToTensor())

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Example loop
    for batch_idx, (images, targets, class_names) in enumerate(dataloader):
        print(f"Batch {batch_idx}: {len(images)} images loaded.")
