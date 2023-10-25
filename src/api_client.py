"""
APIClient's implementation to mimic webportal functionalities
"""

import sys
import argparse
import time
import random

from fedn import APIClient

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="API client to manage fedn tasks.")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address of FEDn controller (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8092,
        help="Listen port of FEDn controller (default: 8092)",
    )
    parser.add_argument(
        "--package",
        type=str,
        required=True,
        help="Path of the required compute package (no default)",
    )
    parser.add_argument(
        "--helper",
        type=str,
        default="pytorchhelper",
        help="Helper used for the compute package (default: pytorchhelper)",
    )
    parser.add_argument(
        "--init_model",
        type=str,
        required=True,
        help="Path of the initial seed model (no default)",
    )
    
    # return parsed arguments
    args = parser.parse_args()
    return args

def main():
    # get system arguments
    args = parse_args()

    # create an instance of the APIClient and connect to FEDn server
    client = APIClient(host=args.host, port=args.port)
    
    # send the compute package and the initial seed model
    client.set_package(args.package, helper=args.helper)
    client.set_initial_model(args.init_model)

    # submit training configurations
    status = False
    while not status:
        # deploy a training round with random session id
        status = client.start_session(
            session_id=f"session_{int(10000000 * random.random())}", 
            helper=args.helper,
            rounds=15,  
            round_timeout=180
        )['success']

        # wait before few seconds before generating new reques
        time.sleep(5)

    # wait for training round to end?
    while client.get_controller_status()["state"] != "idle":
        print ("Waiting for controller to become idle... ")
        time.sleep(15)

    # print train / validation statistics
    # print(client.get_model_trail())
    # print(client.list_validations())

if __name__ == "__main__":
    main()
