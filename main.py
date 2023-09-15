import argparse
import json
import torch
from src.data import Data
from src.train import mcmc_train
from model.AlexNet import AlexNet
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


    model = AlexNet().to(device)
    params = model.parameters()
    trainer = mcmc_train(device=device, params=params, data=data_obj.train_dataloader, config=config)



if __name__ == "__main__":
    main()