import os
import tempfile

import torch
from google.cloud import storage
from torch.utils.data import DataLoader
from torchvision import transforms

BUCKET_NAME = "mlops_bucket100"


class PokeDataset(torch.utils.data.Dataset):
    def __init__(self, bucket_name, mode, transform=None):
        self.bucket_name = bucket_name
        self.mode = mode
        self.transform = transform
        self.file_list = self._get_file_list()
        self.client = None  # Client will be initialized per worker

    def _get_file_list(self):
        # Pre-fetch the list of files from the GCS bucket
        client = storage.Client()  # Temporary client to fetch file list
        bucket = client.bucket(self.bucket_name)
        return [blob.name for blob in bucket.list_blobs(prefix=self.mode)]

    def _get_client(self):
        # Lazily initialize the client in each worker
        if self.client is None:
            self.client = storage.Client()
        return self.client

    def _download_file(self, blob_name):
        tmp_dir = tempfile.gettempdir()
        local_path = os.path.join(tmp_dir, os.path.basename(blob_name))
        if not os.path.exists(local_path):
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
        return local_path

    def __getitem__(self, idx):
        file_path = self._download_file(self.file_list[idx])
        # Load and process the file (replace this with your actual logic)
        data = torch.load(file_path)  # Example: adjust based on your file format
        if self.transform:
            data = self.transform(data)
        target = torch.tensor(0)  # Replace with your target logic
        return data, target

    def __len__(self):
        return len(self.file_list)


def main():
    transform_train = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset and DataLoader
    train_dataset = PokeDataset(BUCKET_NAME, mode="train", transform=transform_train)

    if torch.cuda.is_available():
        num_workers = 6
    else:
        num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        print(f"Batch {batch_idx} processed")


if __name__ == "__main__":
    main()
