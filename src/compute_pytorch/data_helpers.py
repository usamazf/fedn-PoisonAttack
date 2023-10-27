"""Helper functions to handle data loading and splitting."""

import os
from typing import Callable, Optional

import fire

import torch
import numpy as np
from torch.utils.data import Dataset

from common import parse_configs

class CustomDataset(Dataset):
    """ Create a custom dataset with given data and labels
    """
    def __init__(
            self, 
            data, 
            targets, 
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ):
        """ Initialize the dataset
        
        :param data: The data samples of the desired dataset.
        :type data: Union[list, numpy.ndarray, torch.Tensor]
        :param targets: The respective labels of the provided data samples.
        :type targets: Union[C, numpy.ndarray, torch.Tensor]
        :param transform: A function/transform that takes in an PIL image and returns a transformed version.
        :type transform: Optional[Callable]
        :param target_transform: A function/transform that takes in the target and transforms it.
        :type transform: Optional[Callable]
        """
        self.data = data
        if not torch.is_tensor(self.data):
            self.data = torch.tensor(self.data)
        self.data = self.data.float()

        self.targets = targets
        if not torch.is_tensor(self.targets):
            self.targets = torch.tensor(self.targets)
        self.targets = self.targets.long()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """ Fetch a single data sample.
        :param index: Index of the datasample to access.
        :type index: int
        :return: The training and testing splits of the requested dataset.
        :rtype: tuple
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sample, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.targets)


def get_dataset(
        dataset_name: str,
        dataset_path: str,
        dataset_down: bool
    ):
    """ Load the desired dataset.

    :param dataset_name: The name of the dataset to load.
    :type dataset_name: str
    :param dataset_path: The path to load the dataset from.
    :type dataset_path: str
    :param dataset_down: Whether to download the dataset if not available.
    :type dataset_down: bool
    :return: The training and testing splits of the requested dataset.
    :rtype: tuple
    """
    
    assert dataset_name in ["MNIST", "CIFAR-10", "STL-10", "EMNIST-DIGITS", "EMNIST-BYCLASS", "Fashion-MNIST"], f"Invalid dataset {dataset_name} requested."

    if dataset_name == "MNIST":
        # Load MNIST dataset
        from datasets.mnist import load_mnist
        trainset, testset = load_mnist(out_dir=dataset_path, download=dataset_down)
        # Modify data to add extra channel dimension
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1), targets=trainset.targets, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1), targets=testset.targets, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    elif dataset_name == "EMNIST-DIGITS":
        # Load EMNIST dataset
        from datasets.emnist import load_emnist
        trainset, testset = load_emnist(out_dir=dataset_path, download=dataset_down, split="digits")
        # Modify data to add extra channel dimension
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1), targets=trainset.targets, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1), targets=testset.targets, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    elif dataset_name == "EMNIST-BYCLASS":
        from datasets.emnist import load_emnist
        return load_emnist(out_dir=dataset_path, download=dataset_down, split="byclass")
    elif dataset_name == "Fashion-MNIST":
        # Load Fashion-MNIST dataset
        from datasets.fmnist import load_fmnist
        trainset, testset = load_fmnist(out_dir=dataset_path, download=dataset_down)
        # Modify data to add extra channel dimension
        custom_trainset = CustomDataset(data=trainset.data.unsqueeze(1), targets=trainset.targets, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data.unsqueeze(1), targets=testset.targets, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    elif dataset_name == "CIFAR-10":
        # Load CFIAR-10 dataset
        from datasets.cifar10 import load_cifar10
        trainset, testset = load_cifar10(out_dir=dataset_path, download=dataset_down)
        # Modify data to have [S, C, H, W] format
        custom_trainset = CustomDataset(data=trainset.data.transpose((0, 3, 1, 2)), targets=trainset.targets, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data.transpose((0, 3, 1, 2)), targets=testset.targets, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    elif dataset_name == "STL-10":
        # Load STL-10 dataset
        from datasets.stl10 import load_stl10
        trainset, testset = load_stl10(out_dir=dataset_path, download=dataset_down)
        # For some weird reason STL-10 has labels instead 
        # of targets adding additional attribute targets 
        # to make it consistent with other datasets
        custom_trainset = CustomDataset(data=trainset.data, targets=trainset.labels, transform=trainset.transform, target_transform=trainset.target_transform)
        custom_testset = CustomDataset(data=testset.data, targets=testset.labels, transform=testset.transform, target_transform=testset.target_transform)
        return custom_trainset, custom_testset
    else:
        raise Exception(f"Invalid dataset {dataset_name} requested.")


def split_dataset(out_dir: str, config_file: dict):
    """ Split the dataset into chunks.

    :param out_dir: The path where to write the splits.
    :type out_dir: str
    :param config_file: Path of configuration file containing data configs
    :type config_file: str
    """

    # Fetch user configurations
    dataset_configs = parse_configs(config_file)["DATASET_CONFIGS"]

    # Load the requested dataset
    trainset, testset = get_dataset(
        dataset_name=dataset_configs["DATASET_NAME"],
        dataset_path=dataset_configs["DATASET_PATH"],
        dataset_down=dataset_configs["DATASET_DOWN"]
    )

    # Split the dataset based on dirichlet distribution
    trainset_splits = split_dirichlet(
        dataset=trainset,
        num_splits=dataset_configs["N_DATA_SPLIT"],
        dirichlet_alpha=dataset_configs["DIRICHLET_ALPHA"],
        random_seed=dataset_configs["RANDOM_SEED"]
    )
    ## TODO: Split test data among clients as well

    # Make output directory
    if not os.path.exists(f'{out_dir}'):
        os.mkdir(f'{out_dir}')
    if not os.path.exists(f'{out_dir}/splits'):
        os.mkdir(f'{out_dir}/splits')

    # Save the splits
    for i in range(dataset_configs["N_DATA_SPLIT"]):
        subdir = f'{out_dir}/splits/{str(i+1)}'
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        # Save train dataset splits
        torch.save(trainset_splits[i], f"{subdir}/train.pt")

        # Save test dataset
        torch.save([testset[:]], f"{subdir}/test.pt")


def split_dirichlet(dataset, num_splits, dirichlet_alpha, double_stochstic = True, random_seed = 32):
    """Splits data among the workers using dirichlet distribution"""
    
    # Set random seed for reproducable results
    np.random.seed(random_seed)

    labels = dataset.targets
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    n_classes = np.max(labels)+1
    
    # Get label distibution
    label_distribution = np.random.dirichlet([dirichlet_alpha]*num_splits, n_classes)
   
    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() for y in range(n_classes)]
    
    split_idcs = [[] for _ in range(num_splits)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            split_idcs[i] += [idcs]

    split_idcs = [np.concatenate(idcs) for idcs in split_idcs]
    print_split(split_idcs, labels)

    # splits = [[dataset.data[idcs], dataset.targets[idcs]] for idcs in split_idcs]
    splits = [[dataset[idcs]] for idcs in split_idcs]

    return splits


def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    #x = x / x.sum(axis=0).reshape(1,-1)
    return x


def print_split(idcs, labels):
    """Helper function to print data splits."""
    n_labels = np.max(labels) + 1 
    print("Data split:")
    splits = []
    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
        splits += [split]
        if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
            print(" - Split {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
        elif i==len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)
    
    print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
    print()


def load_data(data_path, is_train=True):
    """ Load data from disk. 

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """

    # Load the datasset from disk
    if is_train:
        data = torch.load(f"{data_path}/train.pt")
    else:
        data = torch.load(f"{data_path}/test.pt")

    loaded_dataset = CustomDataset(data[0][0], data[0][1])

    return loaded_dataset


if __name__ == "__main__":
    fire.Fire({
        "split_data": split_dataset,
        # "train": train,
        # "validate": validate,
        # "predict": predict,
    })
