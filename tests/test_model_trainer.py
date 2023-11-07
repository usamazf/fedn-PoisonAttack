"""A testing module to test the data load function."""
import pytest
import os

import sys
sys.path.append(os.path.abspath("src/compute_pytorch"))

import torch
from torch.utils.data import DataLoader

from data_helpers import load_data, split_dataset
from model_helpers import compile_model
from common import get_optimizer, get_criterion, train_model, evaluate_model

# Base dataset configurations
config_base = dict({
    "SERVER_CONFIGS": {
        "N_TRAIN_ROUND": 20,
        "ROUND_TIMEOUT": 250,
    },
    "DATASET_CONFIGS": {
        "DATASET_NAME": "MNIST",
        "DATASET_DOWN": True,
        "DATASET_PATH": "./temp/data",
        "N_DATA_SPLIT": 2,
        "DIRICHLET_ALPHA": 100.0,
        "RANDOM_SEED": 333,
    },
    "MODEL_CONFIGS": {
        "MODEL_NAME": "LENET-1CH",
        "N_CLASSES": 10
    },
    "CLIENT_CONFIGS": {
        "RUN_DEVICE": "auto",
        "LOCAL_EPCH": 1,
        "BATCH_SIZE": 32,
        "LEARN_RATE": 1e-4,
        "OPTIMIZER": "ADAM",
        "CRITERION": "CROSSENTROPY",
    }
})


##################################################
## Test 1: Verify model training
##################################################

def get_client_id():
    # Split the MNIST dataset with base configs first
    split_dataset(out_dir="./temp", user_configs=config_base["DATASET_CONFIGS"])
    
    # Yield individual dataset chunks to test data load
    for client_id in range(config_base["DATASET_CONFIGS"]["N_DATA_SPLIT"]):
        yield client_id+1

@pytest.mark.parametrize("client_id", [x for x in get_client_id()])
def test_model_training(client_id):

    # Check for run device
    local_device = config_base["CLIENT_CONFIGS"]["RUN_DEVICE"]
    if local_device == "auto":
        local_device = f"cuda:{client_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"

    # Get train configurations
    batch_size = config_base["CLIENT_CONFIGS"]["BATCH_SIZE"]
    lr = config_base["CLIENT_CONFIGS"]["LEARN_RATE"]
    epochs = config_base["CLIENT_CONFIGS"]["LOCAL_EPCH"]
    global_rounds = config_base["SERVER_CONFIGS"]["N_TRAIN_ROUND"]

    # Split the MNIST dataset with base configs first
    #split_dataset(out_dir="./temp", user_configs=config_base["DATASET_CONFIGS"])

    # Load the train and test sets
    trainset = load_data(data_path=f"./temp/splits/{client_id}", is_train=True)
    testset = load_data(data_path=f"./temp/splits/{client_id}", is_train=False)

    # Create train loader
    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    # Load model and parameters for training
    model = compile_model(config_base["MODEL_CONFIGS"])
    model = model.to(local_device)

    
    # Prepare optimizer and criterion
    optimizer = get_optimizer(config_base["CLIENT_CONFIGS"]["OPTIMIZER"], model, lr)
    criterion = get_criterion(config_base["CLIENT_CONFIGS"]["CRITERION"])

    train_stats = []
    test_stats = []

    for ge in range(global_rounds):
        # Train the model using train helper module
        train_model(
            model=model, 
            trainloader=trainloader,
            epochs=epochs,
            device=local_device,
            learning_rate=lr,
            criterion=criterion,
            optimizer=optimizer,
            verbose=False
        )

        # Evaluate the model for training data
        train_stats.append(
            evaluate_model(
                model=model,
                dataloader=trainloader,
                device=local_device
            )
        )

        # Evaluate the model for testing data
        test_stats.append(
            evaluate_model(
                model=model,
                dataloader=testloader,
                device=local_device
            )
        )

    print(f"Evaluating Client {client_id}")
    for indx in range(global_rounds):
        print(f"    Train Accuracy = {train_stats[indx]['accuracy']:0.4f}, Test Accuracy = {test_stats[indx]['accuracy']:0.4f}")

    assert (train_stats[0]['accuracy'] < train_stats[-1]['accuracy']) and ((test_stats[0]['accuracy'] < test_stats[-1]['accuracy'])), "[ERROR]: Model accuracy is not increasing."
    
    # Failing on purpose to get verbosity results
    assert False, "This is supposed to fail, so we can check model accuracy."