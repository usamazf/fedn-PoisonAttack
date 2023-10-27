"""Helper functions to handle models."""

import collections

import torch

def compile_model(model_configs: dict) -> torch.nn.Module:
    """ Compile the pytorch model.
    
    :param model_configs: Model configurations.
    :type model_configs: dict
    :return: The compiled model.
    :rtype: torch.nn.Module
    """
    assert model_configs["MODEL_NAME"] in ["SIMPLE-CNN", "SIMPLE-MLP", "LENET-1CH", "LENET-3CH", "RESNET-18"], f"Invalid model {model_configs['MODEL_NAME']} requested."

    if model_configs["MODEL_NAME"] == "SIMPLE-MLP":
        from models.simple_mlp import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "SIMPLE-CNN":
        from models.simple_cnn import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-1CH":
        from models.lenet_1ch import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "LENET-3CH":
        from models.lenet_3ch import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "RESNET-18":
        from models.resnet18 import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    else:
        raise Exception(f"Invalid model {model_configs['MODEL_NAME']} requested.")


def save_model(model, out_path, helper):
    """ Save model to disk. 

    :param model: The model to save.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    param helper: A fedn helper instance.
    :type helper: fedn.utils.helpers.HelperBase
    """
    weights = model.state_dict()
    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy()
    helper.save(weights, out_path)


def load_model(model_path, helper, model_configs) -> torch.nn.Module:
    """ Load model from disk.

    param model_path: The path to load from.
    :type model_path: str
    param helper: A fedn helper instance.
    :type helper: fedn.utils.helpers.HelperBase
    :param model_configs: Model configurations.
    :type data_path: dict
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    weights_np = helper.load(model_path)
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    model = compile_model(model_configs)
    model.load_state_dict(weights)
    model.eval()
    return model
