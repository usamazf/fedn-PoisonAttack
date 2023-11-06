"""A testing module to test the data load function."""
import pytest
import os

import sys
sys.path.append(os.path.abspath("src/compute_pytorch"))

import copy

from data_helpers import load_data, split_dataset

# Base dataset configurations
config_base = dict({
    "DATASET_CONFIGS": {
        "DATASET_NAME": "MNIST",
        "DATASET_DOWN": True,
        "DATASET_PATH": "./temp/data",
        "N_DATA_SPLIT": 2,
        "DIRICHLET_ALPHA": 100.0,
        "RANDOM_SEED": 333,
    }
})


##################################################
## Test 1: Verify dataset loading functionality
##################################################

def get_input_for_load():
    # Split the MNIST dataset with base configs first
    split_dataset(out_dir="./temp", user_configs=config_base["DATASET_CONFIGS"])
    
    # Yield individual dataset chunks to test data load
    for item in os.listdir("./temp/splits"):
        yield f"./temp/splits/{item}"

@pytest.mark.parametrize("data_path", [x for x in get_input_for_load()])
def test_data_load(data_path):
    # Load the train and test sets
    trainset = load_data(data_path=data_path, is_train=True)
    testset = load_data(data_path=data_path, is_train=False)

    # Verify that trainset and testset are successfully loaded
    assert (trainset is not None) and (testset is not None), "Something went wrong while loading data"



##################################################
## Test 2: Verify dataset splitting functionality
##################################################

def get_input_for_split():
    for item in ["MNIST"]: #, "CIFAR-10", "EMNIST"]:
        local_copy = copy.deepcopy(config_base["DATASET_CONFIGS"])
        local_copy["DATASET_NAME"] = item
        yield local_copy

@pytest.mark.parametrize("user_configs", [x for x in get_input_for_split()])
def test_data_split(user_configs):

    split_dataset(out_dir="./temp", user_configs=user_configs)

    assert len(os.listdir("./temp/splits")) > 0, "Something went wrong while splitting the dataset"

