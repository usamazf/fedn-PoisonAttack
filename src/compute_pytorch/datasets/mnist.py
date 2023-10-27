"""A function to load the MNIST digit dataset."""

from typing import Tuple

import torchvision
import torchvision.transforms as transforms


def load_mnist(out_dir="temp/data", download=True) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load MNIST (training and test set)."""
    
    # Define the transform for the data.
    transform = transforms.Compose([
        torchvision.transforms.Resize(size=(32,32), antialias=None),
        # transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.1307,))
    ])

    # Initialize Datasets. MNIST will automatically download if not present
    trainset = torchvision.datasets.MNIST(
        root=out_dir, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=out_dir, train=False, download=download, transform=transform
    )

    # Return the datasets
    return trainset, testset