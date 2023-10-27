"""A function to load the CIFAR-10 dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_cifar10(out_dir="temp/data", download=True) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load CIFAR-10 (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        # transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Initialize Datasets. CIFAR-10 will automatically download if not present
    trainset = torchvision.datasets.CIFAR10(
        root=out_dir, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=out_dir, train=False, download=download, transform=transform
    )
    
    # Return the datasets
    return trainset, testset