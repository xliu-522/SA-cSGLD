import numpy as np
import torch
import random
import math
from scipy.special import expit, logit
from torch import nn
#from torch.func import functional_call, grad
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from types import SimpleNamespace
from src.sampler import SGD, SGLD, SASGLD


class mcmc_train_test(object):
    def __init__(self, device, res_dir, train_data, test_data, config, model):
        self.config = config
        self.device = device
        self.train_dataloader = train_data
        self.test_dataloader = test_data
        self.model = model
        self.sparse = config["sampler"]["sparse"]
        self.sampler_name = self.config["sampler"]["sampler"]
        self.sample_size = config["data"]["sample_size"]
        self.update_rate = config["sampler"]["update_rate"]
        self.total_par = config["model"]["total_par"]
        self.batch_size = config["data"]["batch_size"]
        self.num_batch = self.sample_size//self.batch_size
        self.cycles = config["training"]["cycles"]
        self.epoches = config["training"]["epoches"]
        self.threshold = config["training"]["threshold"]
        self.lr = config["training"]["gamma"]
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.Loss, self.Sparsity, self.Acc = [], [], []
        self.res_dir = res_dir
       
    def train_it(self):
        print("training here!")
        if self.sampler_name == "sasgld":
            self.sampler = SASGLD(device=self.device, config=self.config, model=self.model)
            iters = 1
            for t in range(self.epoches):
                print(f"Epoch {t+1}\n-------------------------------")
                for batch, (X, y) in enumerate(self.train_dataloader):
                    print("Batch: ", batch)
                    X = X.to(self.device)
                    y = y.to(self.device)
                    self.model = self.sampler.sparsify_model_params(self.model)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    self.model = self.sampler.update_params(self.model)
                    self.sampler.select_CoordSet(self.model)
                    self.spar = self.sampler.update_sparsity(self.model)
                    self.model = self.sampler.zero_grad(self.model)
                    if iters % 100 == 0:
                        self.test_it()
                        print('save!')
                        self.model.cpu()
                        torch.save(self.model.state_dict(),f'{self.res_dir}/model_{iters}.pt')
                        self.model.cuda(self.device)
                    self.test_it()
                    iters+=1
        
        elif self.sampler_name == "sacsgld":
            num_iter = self.num_batch*self.epoches
            iter_C = num_iter//self.cycles
            iters = 1
            C = 1
            self.sampler = SASGLD(device=self.device, config=self.config, model=self.model)
            for epoch in range(self.epoches):
                print(f"Epoch {epoch+1}\n-------------------------------")
                for batch, (X, y) in enumerate(self.train_dataloader):
                    print("Batch: ", batch)
                    X = X.to(self.device)
                    y = y.to(self.device)
                    r_k = ((k-1) % iter_C)/iter_C
                    if r_k == 0:
                        print('save!')
                        self.model.cpu()
                        torch.save(self.model.state_dict(),f'{self.res_dir}/model_cycle{C}.pt')
                        C += 1
                        self.model.cuda(self.device)
                    a_k = max(self.lr/2 * (math.cos(math.pi * r_k) + 1), self.threshold)
                    print('r_k: ', r_k)
                    print('a_k: ', a_k)
                    self.model = self.sampler.sparsify_model_params(self.model)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    self.model = self.sampler.update_params(self.model)
                    self.sampler.select_CoordSet(self.model)
                    self.spar = self.sampler.update_sparsity(self.model)
                    self.model = self.sampler.zero_grad(self.model)
                    if a_k == 0:
                        self.model = self.sampler.reinit_sparsity(self.model)
                        
                    if iters % 100 == 0:
                        self.test_it()
                    iters+=1

        elif self.sampler_name == "csgld":
            num_iter = self.num_batch*self.epoches
            iter_C = num_iter//self.cycles
            iters = 1
            C = 1
            self.sampler = SASGLD(device=self.device, config=self.config, model=self.model)
            for epoch in range(self.epoches):
                print(f"Epoch {epoch+1}\n-------------------------------")
                for batch, (X, y) in enumerate(self.train_dataloader):
                    print("Batch: ", batch)
                    X = X.to(self.device)
                    y = y.to(self.device)
                    r_k = ((iters-1) % iter_C)/iter_C
                    if r_k == 0:
                        print('save!')
                        self.model.cpu()
                        torch.save(self.model.state_dict(),f'{self.res_dir}/model_cycle{C}.pt')
                        C += 1
                        self.model.cuda(self.device)
                    a_k = max(self.lr/2 * (math.cos(math.pi * r_k) + 1), self.threshold)
                    print('r_k: ', r_k)
                    print('a_k: ', a_k)
                    # Do SGLD
                    params = self.model.parameters()
                    optimizer = SGLD(params, a_k, self.sample_size)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if iters % 100 == 0:
                        self.test_it()
                    iters+=1

            
        elif self.sampler_name == "sgld":
            params = self.model.parameters()
            optimizer = SGLD(params, self.lr, self.sample_size)
            iters = 1
            for t in range(self.epoches):
                print(f"Epoch {t+1}\n-------------------------------")
                for batch, (X, y) in enumerate(self.train_dataloader):
                    X = X.to(self.device) 
                    y = y.to(self.device)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if iters % 100 == 0:
                        self.test_it()
                        print('save!')
                        self.model.cpu()
                        torch.save(self.model.state_dict(),f'{self.res_dir}/model_{iters}.pt')
                        self.model.cuda(self.device)
                    iters+=1
            
        return self.Loss, self.Acc, self.Sparsity
                    
                

    def test_it(self):
        test_size=len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        if self.sparse:
            self.model = self.sampler.sparsify_model_params(self.model)
        with torch.no_grad():
            test_loss = 0
            correct = 0
            for X, y in self.test_dataloader:
                X = X.to(self.device) 
                y = y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
        test_loss /= num_batches
        correct /= test_size
        self.Loss.append(test_loss)
        self.Acc.append(correct)
        if self.sparse:
            self.Sparsity.append(self.spar)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
