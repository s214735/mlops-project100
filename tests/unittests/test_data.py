import numpy as np
from p100.data2 import PokeDataset
from torch.utils.data import DataLoader, Dataset

BUCKET_NAME = "mlops_bucket100"


def test_my_dataset():
    """Test the MyDataset class."""
    dataset_train = PokeDataset(BUCKET_NAME, mode="train")
    dataset_val = PokeDataset(BUCKET_NAME, mode="val")
    dataset_test = PokeDataset(BUCKET_NAME, mode="test")

    train_dataloader = DataLoader(dataset_train, batch_size=1)
    val_dataloader = DataLoader(dataset_val, batch_size=1)
    test_dataloader = DataLoader(dataset_test, batch_size=1)

    img_train, target_train, _ = next(iter(train_dataloader))
    img_val, target_val, _ = next(iter(val_dataloader))
    img_test, target_test, _ = next(iter(test_dataloader))

    # Check data class
    assert isinstance(dataset_train, Dataset)

    # Check data shape
    assert img_train.shape == (1, 3, 128, 128)

    # Check length of dataset
    assert len(train_dataloader) + len(val_dataloader) + len(test_dataloader) == 26539

    # Check if all classes are represented
    assert len(np.unique(train_dataloader.dataset.targets)) == 1000


if __name__ == "__main__":
    test_my_dataset()
