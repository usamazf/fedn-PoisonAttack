"""A function to load the MNIST digit dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_emnist(out_dir="temp/data", download=True, split="mnist") -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load EMNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        torchvision.transforms.Resize(size=(32,32), antialias=None),
        # transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.1307,))
    ])

    # Initialize Datasets. EMNIST will automatically download if not present
    trainset = torchvision.datasets.EMNIST(
        root=out_dir, train=True, split=split, download=download, transform=transform
    )
    testset = torchvision.datasets.EMNIST(
        root=out_dir, train=False, split=split, download=download, transform=transform
    )

    # Return the datasets
    return trainset, testset