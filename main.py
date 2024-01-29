import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import json
import torch
from torch import nn
from src.data import Data
from src.train import mcmc_train_test
import models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
    
    # Create a random dummy batch with the specified shape for Alexnet
    if model_name == "AlexNet":
        dummy_batch = torch.randn(config["data"]["batch_size"], config["data"]["num_channels"], config["data"]["image_size"], config["data"]["image_size"]).to(device)
        model(dummy_batch)

    config["model"]["total_par"] = sum(P.numel() for P in model.parameters() if P.requires_grad)
    print(config["model"]["total_par"])
    
    
    print("**** Create directory ****")
    # Get the current date and time
    current_time = datetime.now()

    # Format the current time as a string (adjust the format as needed)
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    res_dir = f'./result/{config["sampler"]["sampler"]}_{config["model"]["model_name"]}_{config["data"]["dataset"]}_{config["training"]["gamma"]}_{config["training"]["epoches"]}_{config["training"]["cycles"]}_{time_string}'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        print(f"Folder '{res_dir}' created.")
    else:
        print(f"Folder '{res_dir}' already exists.")

    trainer = mcmc_train_test(device=device, res_dir=res_dir, train_data=data_obj.train_dataloader, test_data = data_obj.test_dataloader, config=config, model=model)
    
    print("**** Start training ****")
    loss, acc, sparsity= trainer.train_it()
    
    np.save(f'{res_dir}/loss.npy',loss)
    np.save(f'{res_dir}/accuracy.npy', acc)
    
    
    plt.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Testing Loss Curve')
    plt.savefig(f'{res_dir}/loss.png')
    plt.close()
    plt.plot(acc)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Testing accuracy Curve')
    plt.savefig(f'{res_dir}/accuracy.png')
    plt.close()
    if config["sampler"]["sampler"] == "sasgld" or config["sampler"]["sampler"] == "sacsgld":
        plt.plot(sparsity)
        plt.xlabel('Iterations')
        plt.ylabel('Sparsity')
        plt.title('Sparsity Curve')
        plt.savefig(f'./{res_dir}/sparsity.png')
        np.save(f'{res_dir}/sparsity.npy', sparsity)
    plt.close()
    



if __name__ == "__main__":
    main()