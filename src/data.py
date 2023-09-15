import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from types import SimpleNamespace

class Data(object):
    def __init__(self, config):
        self.data_path = config['data']['data_path']
        self.batch_size =config['data']['batch_size']
        self.image_size = config['data']['image_size']

    def read(self):
        resize_transform = transforms.Resize(size=(self.image_size,self.image_size),antialias=True)
        train_dataset = datasets.CIFAR10(root=self.data_path, train=True, download=True)
        self.data_mean = (train_dataset.data / 255.0).mean(axis=(0,1,2))
        self.data_std = (train_dataset.data / 255.0).std(axis=(0,1,2))
        print(self.data_mean)
        data_transform = transforms.Compose([transforms.ToTensor(), resize_transform,transforms.Normalize(self.data_mean, self.data_std)])

        # Loading the training dataset. 
        self.train_dataset = datasets.CIFAR10(root=self.data_path, train=True, transform=data_transform, download=True)
        # Loading the test set
        self.test_dataset = datasets.CIFAR10(root=self.data_path, train=False, transform=data_transform, download=True)

        # We define a set of data loaders that we can use for various purposes later.
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=4)
        