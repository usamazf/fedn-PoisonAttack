"""A testing module to test the data load function."""
import pytest
import os

import sys
sys.path.append(os.path.abspath("src/compute_pytorch"))

import copy
import torch

from data_helpers import load_data, split_dataset
from model_helpers import compile_model

# Base dataset configurations
config_base = dict({
    "MODEL_CONFIGS": {
        "MODEL_NAME": "SIMPLE-MLP",
        "NUM_CLASSES": 10
    }
})


##################################################
## Test 1: Verify model compilations
##################################################

def get_input_for_compile():
    # Yield all model types
    for item in ["SIMPLE-CNN", "SIMPLE-MLP", "LENET-1CH", "LENET-3CH", "RESNET-18"]:
        local_copy = copy.deepcopy(config_base["MODEL_CONFIGS"])
        local_copy["MODEL_NAME"] = item
        yield local_copy

@pytest.mark.parametrize("model_configs", [x for x in get_input_for_compile()])
def test_model_load(model_configs):
    model = compile_model(model_configs)
    # Verify that trainset and testset are successfully loaded
    assert isinstance(model, torch.nn.Module), "Something went wrong while loading model"



##################################################
## Test 2: Verify model forward-backward pass
##################################################

def test_model_forward():
    rand_input = torch.rand([128, 1, 32, 32])
    model = compile_model(config_base["MODEL_CONFIGS"])
    output = model(rand_input)

    # Verify that trainset and testset are successfully loaded
    assert (output.sum() - 1.0) >= 1e-12, "Something went wrong while doing a forward pass"
