import fire
import torch
from torch.utils.data import DataLoader

from fedn.utils.helpers import get_helper, save_metadata, save_metrics
from model_helpers import save_model, load_model, compile_model
from data_helpers import load_data
from common import parse_configs, train_model, evaluate_model, get_criterion, get_optimizer

HELPER_MODULE = 'pytorchhelper'
NUM_CLASSES = 10

# def _get_data_path():
#     """ For test automation using docker-compose. """
#     # Figure out FEDn client number from container name
#     # client = docker.from_env()
#     # container = client.containers.get(os.environ['HOSTNAME'])
#     # number = container.name[-1]

#     # Return data path
#     # return f"/var/data/clients/{number}/mnist.pt"
#     return f"/var/data/"

def init_seed(out_path, config_file="config.yaml"):
    """ Initialize seed model.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Get FEDn helper module
    helper = get_helper(HELPER_MODULE)

    # Fetch user configurations
    configs = parse_configs(config_file)
    
    # Initialize and save the model
    model = compile_model(configs["MODEL_CONFIGS"])
    save_model(model, out_path, helper)


def train(in_model_path, out_model_path, data_path=None, config_file="config.yaml"):
    """ Train model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param config_file: Current run configurations.
    :type config_file: dict
    """
    # Fetch user configurations
    configs = parse_configs(config_file)

    # Check for run device
    local_device = configs["CLIENT_CONFIGS"]["RUN_DEVICE"]
    if local_device == "auto":
        local_device = f"cuda" if torch.cuda.is_available() else "cpu"

    # Get FEDn helper module
    helper = get_helper(HELPER_MODULE)

    # Get train configurations
    batch_size = configs["CLIENT_CONFIGS"]["BATCH_SIZE"]
    lr = configs["CLIENT_CONFIGS"]["LEARN_RATE"]
    epochs = configs["CLIENT_CONFIGS"]["LOCAL_EPCH"]

    # Load dataset for training
    trainset = load_data(data_path)

    # Create train loader
    trainloader = DataLoader(trainset, batch_size=batch_size)

    # Load model and parameters for training
    model = load_model(in_model_path, helper, configs["MODEL_CONFIGS"])
    model = model.to(local_device)

    
    # Prepare optimizer and criterion
    optimizer = get_optimizer(configs["CLIENT_CONFIGS"]["OPTIMIZER"], model, lr)
    criterion = get_criterion(configs["CLIENT_CONFIGS"]["CRITERION"])

    # Train the model using train helper module
    train_model(
        model=model, 
        trainloader=trainloader,
        epochs=epochs,
        device=local_device,
        learning_rate=lr,
        criterion=criterion,
        optimizer=optimizer
    )

    # Metadata needed for aggregation server side
    metadata = {
        'num_examples': len(trainset),
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr
    }

    # Save JSON metadata file
    save_metadata(metadata, out_model_path)

    # Save model update
    save_model(model, out_model_path, helper)


def validate(in_model_path, out_json_path, data_path=None, config_file="config.yaml"):
    """ Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Fetch user configurations
    configs = parse_configs(config_file)

    # Check for run device
    local_device = configs["CLIENT_CONFIGS"]["RUN_DEVICE"]
    if local_device == "auto":
        local_device = f"cuda" if torch.cuda.is_available() else "cpu"

    # Get FEDn helper module
    helper = get_helper(HELPER_MODULE)

    # Get train configurations
    batch_size = configs["CLIENT_CONFIGS"]["BATCH_SIZE"]

    # Load dataset for training
    trainset = load_data(data_path)
    testset = load_data(data_path, is_train=False)

    # Create train and test dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    # Load model and parameters for training
    model = load_model(in_model_path, helper, configs["MODEL_CONFIGS"])
    model = model.to(local_device)

    # Evaluate the model for training data
    train_stats = evaluate_model(
        model=model,
        dataloader=trainloader,
        device=local_device
    )
    
    # Evaluate the model for testing data
    test_stats = evaluate_model(
        model=model,
        dataloader=testloader,
        device=local_device
    )
    
    # JSON schema
    report = {
        "training_loss": train_stats["loss"],
        "training_accuracy": train_stats["accuracy"],
        "test_loss": test_stats["loss"],
        "test_accuracy": test_stats["accuracy"],
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    fire.Fire({
        "init_seed": init_seed,
        "train": train,
        "validate": validate,
        # "predict": predict,
    })
