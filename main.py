import argparse
import json
import torch
from src.data import Data
from src.train import mcmc_train
#import importlib
#import model.models
import models
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

    model = model.to(device)
    params = model.parameters()
    trainer = mcmc_train(device=device, params=params, data=data_obj.train_dataloader, config=config)
    trainer.train_it()



if __name__ == "__main__":
    main()