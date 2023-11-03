"""APIClient's implementation to mimic webportal functionalities."""

import fire

import time
import random

from fedn import APIClient
from compute_pytorch.common import parse_configs

def submit_compute_configs(package, init_model, host="localhost", port=8092, helper="pytorchhelper", retries=10):
    while True:
        try:
            retries -= 1
            
            # Log message
            print("Trying to submit compute package and seed model...")
            
            # Create an instance of the APIClient and connect to FEDn server
            client = APIClient(host=host, port=port)
            
            # Send the compute package and the initial seed model
            client.set_package(package, helper=helper)
            client.set_initial_model(init_model)

            # Log message
            print("Successfully submitted compute package and seed model")
            break
        
        except:
            if retries <= 0: break
            # FEDn network might not have started
            time.sleep(30)


def submit_train_request(config_file, host="localhost", port=8092, helper="pytorchhelper"):
    
    # Fetch user configurations
    exp_configs = parse_configs(config_file)

    # Create an instance of the APIClient and connect to FEDn server
    client = APIClient(host=host, port=port)
    
    # Submit training configurations
    status = False
    while not status:
        # Deploy a training round with random session id
        status = client.start_session(
            session_id=f"session_{int(10000000 * random.random())}", 
            helper=helper,
            rounds=exp_configs["SERVER_CONFIGS"]["N_TRAIN_ROUND"],
            round_timeout=exp_configs["SERVER_CONFIGS"]["ROUND_TIMEOUT"]
        )['success']

        # Wait before few seconds before generating new request
        time.sleep(5)

def fetch_compute_results():
    pass

# def main():
#     # Get system arguments
#     args = parse_args()

#     # Wait for training round to end?
#     while client.get_controller_status()["state"] != "idle":
#         print ("Waiting for controller to become idle... ")
#         time.sleep(15)

#     # Print train / validation statistics
#     # print(client.get_model_trail())
#     # print(client.list_validations())

if __name__ == "__main__":
    fire.Fire({
        "submit": submit_compute_configs,
        "train": submit_train_request,
        "report": fetch_compute_results,
        # "validate": validate,
        # "predict": predict,
    })
