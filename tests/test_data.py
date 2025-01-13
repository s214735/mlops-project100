from p100.data import MyDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


def test_my_dataset():
    """Test the MyDataset class."""
    dataset_train = MyDataset("data/processed", mode = "train", transform=transforms.ToTensor())
    dataset_val = MyDataset("data/processed", mode = "val", transform=transforms.ToTensor())
    dataset_test = MyDataset("data/processed", mode = "test", transform=transforms.ToTensor())

    data, target_val, class_name = DataLoader(dataset_val, batch_size=1, shuffle=True)
    data, target_test, class_name = DataLoader(dataset_test, batch_size=1, shuffle=True)
    data, target_train, class_name = DataLoader(dataset_train, batch_size=1, shuffle=True)
    # Check data class
    assert isinstance(dataset_train, Dataset)
    # Check data shape
    assert data.shape == (1, 3, 128, 128)
    # Check length of dataset
    assert len(target_train) + len(target_val) + len(target_test) == 26539
    # Check if all classes are represented
    target_train = torch.unique(target_train)
    assert (target_train == torch.arange(0,999)).all()

if __name__ == "__main__":
    test_my_dataset()
