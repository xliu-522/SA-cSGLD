import numpy as np
import torch
import random
from scipy.special import expit, logit
from torch import nn
#from torch.func import functional_call, grad
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from types import SimpleNamespace
from src.sacsgld import sasgldSampler, sacsgldSampler

class mcmc_train_test(object):
    def __init__(self, device, train_data, test_data, config, model):
        self.config = config
        self.device = device
        self.train_dataloader = train_data
        self.test_dataloader = test_data
        self.model = model
        self.sample_size = config["data"]["sample_size"]
        self.update_rate = config["sampler"]["update_rate"]
        self.total_par = config["model"]["total_par"]
        self.batch_size = config["data"]["batch_size"]
        self.cycles = config["sampler"]["C"]
        self.epoches = config["training"]["epoches"]
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def train_it(self):
        print("training here!")
        if config["sampler"]["sampler"] == "sasgld":
            self.sampler = sasgldSampler(device=self.device, config=self.config, model=self.model)
            for t in range(self.epoches):
                print(f"Epoch {t+1}\n-------------------------------")
                for batch, (X, y) in enumerate(self.train_dataloader):
                    print("Batch: ", batch)
                    X = X.to(self.device)
                    y = y.to(self.device)
                    self.sampler.sparsify_model_params()
                    self.sampler.grad_loss(X, y)
                    self.sampler.update_params()
                    self.sampler.select_CoordSet()
                    self.sampler.update_sparsity()
                    self.sampler.zero_grad()
        elif config["sampler"]["sampler"] == "sacsgld":
            self.sampler = sacsgldSampler(device=self.device, config=self.config, model=self.model)

    def test_it(self):
        test_size=len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.sampler.sparsify_model_params()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            for X, y in self.test_dataloader:
                X = X.to(self.device) 
                y = y.to(self.device)
                pred = self.sampler.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
        test_loss /= num_batches
        correct /= test_size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
