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
import time

class mcmc_train_test(object):
    def __init__(self, device, res_dir, train_data, test_data, config, model):
        self.config = config
        self.device = device
        self.train_dataloader = train_data
        self.test_dataloader = test_data
        self.model = model
        self.data_path = config['data']['data_path']
        self.image_size = config['data']['image_size']
        self.model_name = config['model']['model_name']
        self.sampler_name = self.config["sampler"]["sampler"]
        self.sample_size = config["data"]["sample_size"]
        self.update_rate = config["sampler"]["update_rate"]
        #self.total_par = config["model"]["total_par"]
        self.batch_size = config["data"]["batch_size"]
        self.num_batch = self.sample_size//self.batch_size
        self.cycles = config["training"]["cycles"]
        self.numIter = config["training"]["numIter"]
        self.epoches = config["training"]["epoches"]
        self.threshold = config["training"]["threshold"]
        self.lr0 = config["training"]["gamma"]
        self.alpha_0 = config["training"]["alpha_0"]
        self.loss_fn = nn.CrossEntropyLoss()
        self.Loss, self.Sparsity, self.Acc = [], [], []
        self.res_dir = res_dir
       
    def train_it(self):
        print("training here!")
        if self.sampler_name == "sasgld":
            self.sampler = SASGLD(device=self.device, config=self.config, model=self.model)
            iters = 1
            start_time = time.perf_counter()
            for t in range(self.epoches):
                print(f"Epoch {t+1}\n-------------------------------")
                for batch, (X, y) in enumerate(self.train_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    self.model = self.sampler.sparsify_model_params(self.model)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    self.model = self.sampler.update_params(self.model, self.lr0)
                    self.sampler.select_CoordSet(self.model)
                    
                    # calculate 2nd gradient calculate
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    self.spar = self.sampler.update_sparsity(self.model)
                    self.model = self.sampler.zero_grad(self.model)
                    # self.sampler.select_CoordSet(self.model)
                    # self.model = self.sampler.sparsify_model_params(self.model)
                    # pred = self.model(X)
                    # loss = self.loss_fn(pred, y)
                    # loss.backward()
                    # self.model = self.sampler.update_params(self.model, self.lr0)
                    # self.spar = self.sampler.update_sparsity(self.model)
                    # self.model = self.sampler.zero_grad(self.model)
                    if iters % 100 == 0:
                        self.test_it()
                    if iters % 10000 == 0:
                        print('save!')
                        self.model.cpu()
                        torch.save(self.model.state_dict(),f'{self.res_dir}/model_{iters}.pt')
                        self.model.to(self.device)
                    iters+=1
            end_time = time.perf_counter()
            running_time = end_time - start_time
            print(f"Running time: {running_time} seconds")
            with open(f"{self.res_dir}/running_time.txt", "w") as file:
                file.write(f"Running time: {running_time} seconds")
            
                    
        
        elif self.sampler_name == "sacsgld":
            num_iter = self.num_batch*self.epoches
            cyclelen = num_iter//self.cycles
            kk = 1
            C = 1
            self.sampler = SASGLD(device=self.device, config=self.config, model=self.model)
            start_time = time.perf_counter()
            for epoch in range(self.epoches):
                print(f"Epoch {epoch+1}\n-------------------------------")
                for batch, (X, y) in enumerate(self.train_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)
                    r = ((kk-1) % cyclelen)/cyclelen
                    alpha = max(self.alpha_0/2 * (math.cos(2*math.pi * r) + 1), self.threshold)
                    lr = self.lr0*alpha
                    if r == 0:
                        self.model = self.sampler.reinit_sparsity(self.model)
                    if r < 0.0:
                        self.model = self.sampler.exploration(self.model)
                        self.spar = self.sampler.calculate_sparsity(self.model)
                    else:
                        self.model = self.sampler.sparsify_model_params(self.model)
                        pred = self.model(X)
                        loss = self.loss_fn(pred, y)
                        loss.backward()
                        self.model = self.sampler.update_params(self.model, lr)
                        self.sampler.select_CoordSet(self.model)

                        # calculate 2nd gradient calculate
                        pred = self.model(X)
                        loss = self.loss_fn(pred, y)
                        loss.backward()
                        self.spar = self.sampler.update_sparsity(self.model)
                        self.model = self.sampler.zero_grad(self.model)
                    if r == 0 and kk != 1:
                        print('save!')
                        self.model.cpu()
                        torch.save(self.model.state_dict(),f'{self.res_dir}/model_cycle{C}.pt')
                        C += 1
                        self.model.to(self.device)
                        
                    if kk % 100 == 0:
                        self.test_it()
                        print('r: ', r)
                        print('alpha: ', alpha)
                        print('lr:', lr)
                    kk+=1
            end_time = time.perf_counter()
            running_time = end_time - start_time
            print(f"Running time: {running_time} seconds")
            with open(f"{self.res_dir}/running_time.txt", "w") as file:
                file.write(f"Running time: {running_time} seconds")

        elif self.sampler_name == "csgld":
            num_iter = self.num_batch*self.epoches
            cyclelen = num_iter//self.cycles
            kk = 1
            C = 1
            #self.sampler = SASGLD(device=self.device, config=self.config, model=self.model)
            start_time = time.perf_counter()
            for epoch in range(self.epoches):
                print(f"Epoch {epoch+1}\n-------------------------------")
                for batch, (X, y) in enumerate(self.train_dataloader):
                    # print("Batch: ", batch)
                    X = X.to(self.device)
                    y = y.to(self.device)
                    r = ((kk-1) % cyclelen)/cyclelen
                    alpha = max(self.alpha_0/2 * (math.cos(math.pi * r) + 1), self.threshold)
                    lr = self.lr0*alpha
                    if r == 0 and kk != 1:
                        print('save!')
                        self.model.cpu()
                        torch.save(self.model.state_dict(),f'{self.res_dir}/model_cycle{C}.pt')
                        C += 1
                        self.model.to(self.device)
                    
                    # Do SGLD
                    params = self.model.parameters()
                    optimizer = SGLD(params, lr, self.lr0, self.sample_size)
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if kk % 100 == 0:
                        self.test_it()
                        print('r: ', r)
                        print('alpha: ', alpha)
                        print('lr:', lr)
                    kk+=1
            end_time = time.perf_counter()
            running_time = end_time - start_time
            print(f"Running time: {running_time} seconds")
            with open(f"{self.res_dir}/running_time.txt", "w") as file:
                file.write(f"Running time: {running_time} seconds")
            
        elif self.sampler_name == "sgld":
            params = self.model.parameters()
            optimizer = SGLD(params, self.lr0, self.lr0, self.sample_size)
            kk = 1
            start_time = time.perf_counter()
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
                    if kk % 100 == 0:
                        self.test_it()
                    if kk % 1950 == 0:
                        print('save!')
                        self.model.cpu()
                        torch.save(self.model.state_dict(),f'{self.res_dir}/model_{kk}.pt')
                        self.model.to(self.device)
                    kk +=1
            end_time = time.perf_counter()
            running_time = end_time - start_time
            print(f"Running time: {running_time} seconds")
            with open(f"{self.res_dir}/running_time.txt", "w") as file:
                file.write(f"Running time: {running_time} seconds")
        return self.Loss, self.Acc, self.Sparsity
                    
                

    def test_it(self):
        test_size=len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        if self.sampler_name == "sasgld" or self.sampler_name == "sacsgld":
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
        if self.sampler_name == "sasgld" or self.sampler_name == "sacsgld":
            self.Sparsity.append(self.spar)
            print(f"Sparsity: {self.spar} \n")
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")