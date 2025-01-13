import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))
from p100.data import PokeDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


def test_my_dataset():
    """Test the MyDataset class."""
    dataset_train = PokeDataset(mode = "train")
    dataset_val = PokeDataset(mode = "val")
    dataset_test = PokeDataset(mode = "test")

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
    target_train = torch.unique(target_train)
    print(dir(val_dataloader.targets))
    #assert (target_train == torch.arange(0,999)).all()
    
if __name__ == "__main__":
    test_my_dataset()
