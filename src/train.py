import numpy as np
import torch
import random
from scipy.special import expit, logit
from torch import nn
from torch.func import functional_call, grad
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from types import SimpleNamespace

class mcmc_train(object):
    def __init__(self, device, params, data, config):
        self.config = config
        self.device = device
        self.lr = torch.tensor(config["sampler"]["learning_rate"]).to(self.device)
        print(self.lr)
        self.sample_size = config["data"]["spleSize"]
        self.update_rate = config["sampler"]["update_rate"]
    
    def train_it(self):
        print("trainig here!")

