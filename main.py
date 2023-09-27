import argparse
import json
import torch
from torch import nn
from src.data import Data
from src.train import mcmc_train_test
import models
import numpy as np
#from unique_names_generator import get_random_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs='?', type=str, default="config/config.json", required=False)
    args = parser.parse_args()
    try:
        with open(args.config, ) as config:
            config = json.load(config)
            #config['model']["random_name"] = get_random_name().replace(" ", "_")
    except:
        print("Error in config")
    
    print("**** Checking device ****")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device: ", device)

    print("**** Reading data ****")
    data_obj = Data(config)
    data_obj.read()
    config["data"]["sample_size"] = len(data_obj.train_dataset)

    print("**** Load Model ****")
    model_name = config["model"]["model_name"]
    model_reference = getattr(models, model_name)
    if model_reference is not None and callable(model_reference):
        # Create an instance of the class
        model = model_reference()
        print(f"Found model {model_name}. ")
    else:
        print(f"Model {model_name} not found.")

    # Initialize the neural network with a random dummy batch (Lazy)
    model = model.to(device)
    # Create a random dummy batch with the specified shape
    dummy_batch = torch.randn(config["data"]["batch_size"], config["data"]["num_channels"], config["data"]["image_size"], config["data"]["image_size"]).to(device)
    model(dummy_batch)

    config["model"]["total_par"] = sum(P.numel() for P in model.parameters() if P.requires_grad)
    print(config["model"]["total_par"])

    trainer = mcmc_train_test(device=device, train_data=data_obj.train_dataloader, test_data = data_obj.test_dataloader, config=config, model=model)
    
    print("**** Start training ****")
    trainer.train_it()
    trainer.test_it()



if __name__ == "__main__":
    main()