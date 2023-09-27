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

class sasgldSampler(object):
    def __init__(self, device, config, model):
        self.device = device
        self.model = model
        self.lr = torch.tensor(config["sampler"]["learning_rate"], dtype=torch.float32).to(self.device)
        self.gamma = torch.tensor(config["training"]["gamma"], dtype=torch.float32).to(self.device)
        self.rho_0 = torch.tensor(config["sampler"]["rho_0"], dtype=torch.float32).to(self.device)
        self.rho_1 = torch.tensor(config["sampler"]["rho_1"], dtype=torch.float32).to(self.device)
        self.spleSize = config["data"]["sample_size"]
        self.u = torch.tensor(config["sampler"]["u"], dtype=torch.float32).to(self.device)
        self.total_par = torch.tensor(config["model"]["total_par"], dtype=torch.float32).to(self.device)
        self.update_rate = config["sampler"]["update_rate"]
        self.batch_size = config["data"]["batch_size"]
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        # Initiate parameter structure
        self.params_struct, self.J_vec, self.CoordSet  = {}, {}, {}
        i = 0
        for P in self.model.parameters():
            if P.requires_grad:
                self.params_struct[i] = torch.ones(size=P.shape)
                self.J_vec[i] = torch.ceil(self.update_rate*torch.prod(torch.tensor(P.shape)))
                self.CoordSet[i] = torch.zeros(self.J_vec[i].numel())
            i+=1
        
        self.a = self.u * torch.log(self.total_par) + 0.5*torch.log(self.rho_0/self.rho_1)
    

    def zero_grad(self):
        ## Set gradients of all parameters to zero
        for P in self.model.parameters():
            if P.grad is not None:
                P.grad.detach_() # For second-order optimizers important
                P.grad.zero_()
                
                
    def grad_loss(self, X, y):
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        loss.backward()

    def select_CoordSet(self):
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            self.CoordSet[i] = np.random.choice(a=P.numel(), size = int(self.J_vec[i].item()), replace = False)
            i+=1


    @torch.no_grad()
    def update_params(self):
        ## Update all parameters
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            gauss = torch.distributions.Normal(torch.zeros_like(P.data), torch.ones_like(P.data)).sample()
            P[self.params_struct[i]==0].data = torch.sqrt(1/self.rho_0)*gauss[self.params_struct[i]==0]
            index = self.params_struct[i]==1
            g = - self.gamma * ( self.rho_1*P.data[index] + self.spleSize*P.grad[index]) + torch.sqrt(2*self.gamma)*gauss[index]
            #g = -self.spleSize*self.gamma * P.grad[index] + torch.sqrt(2*self.gamma)*gauss[index]
            P.data[index]+=g
            i+=1
            

    @torch.no_grad()
    def update_sparsity(self):
        ## Update all selected parameters structures
        i=0
        sparse_sum = 0.0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            Grad = P.grad.reshape(-1)[self.CoordSet[i]]
            Weights = P.data.reshape(-1)[self.CoordSet[i]]
            zz1 = self.a + 0.5*(self.rho_1 - self.rho_0)*Weights**2 -self.spleSize * -Grad*Weights
            prob = torch.sigmoid(-zz1)
            bern = torch.distributions.Binomial(1, prob).sample()
            params_struct_temp = self.params_struct[i].reshape(-1).to(self.device)
            params_struct_temp[self.CoordSet[i]] = bern
            self.params_struct[i] = params_struct_temp.reshape(self.params_struct[i].shape)
            sparse_sum += torch.sum(self.params_struct[i])
            i+=1
        print("sparsity: ", sparse_sum)
    
    
    def sparsify_model_params(self):
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            weights = P.data*self.params_struct[i].to(self.device)
            P.data = nn.parameter.Parameter(weights)
            i+=1
            
class sacsgldSampler(object):
    def __init__(self, device, config, model):
        self.device = device
        self.model = model
        self.lr = torch.tensor(config["sampler"]["learning_rate"], dtype=torch.float32).to(self.device)
        self.gamma = torch.tensor(config["training"]["gamma"], dtype=torch.float32).to(self.device)
        self.rho_0 = torch.tensor(config["sampler"]["rho_0"], dtype=torch.float32).to(self.device)
        self.rho_1 = torch.tensor(config["sampler"]["rho_1"], dtype=torch.float32).to(self.device)
        self.spleSize = config["data"]["sample_size"]
        self.u = torch.tensor(config["sampler"]["u"], dtype=torch.float32).to(self.device)
        self.total_par = torch.tensor(config["model"]["total_par"], dtype=torch.float32).to(self.device)
        self.update_rate = config["sampler"]["update_rate"]
        self.batch_size = config["data"]["batch_size"]
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        # Initiate parameter structure
        self.params_struct, self.J_vec, self.CoordSet  = {}, {}, {}
        i = 0
        for P in self.model.parameters():
            if P.requires_grad:
                self.params_struct[i] = torch.ones(size=P.shape)
                self.J_vec[i] = torch.ceil(self.update_rate*torch.prod(torch.tensor(P.shape)))
                self.CoordSet[i] = torch.zeros(self.J_vec[i].numel())
            i+=1
        
        self.a = self.u * torch.log(self.total_par) + 0.5*torch.log(self.rho_0/self.rho_1)
    

    def zero_grad(self):
        ## Set gradients of all parameters to zero
        for P in self.model.parameters():
            if P.grad is not None:
                P.grad.detach_() # For second-order optimizers important
                P.grad.zero_()
                
                
    def grad_loss(self, X, y):
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        loss.backward()

    def select_CoordSet(self):
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            self.CoordSet[i] = np.random.choice(a=P.numel(), size = int(self.J_vec[i].item()), replace = False)
            i+=1


    @torch.no_grad()
    def update_params(self):
        ## Update all parameters
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            gauss = torch.distributions.Normal(torch.zeros_like(P.data), torch.ones_like(P.data)).sample()
            P[self.params_struct[i]==0].data = torch.sqrt(1/self.rho_0)*gauss[self.params_struct[i]==0]
            index = self.params_struct[i]==1
            g = - self.gamma * ( self.rho_1*P.data[index] + self.spleSize*P.grad[index]) + torch.sqrt(2*self.gamma)*gauss[index]
            #g = -self.spleSize*self.gamma * P.grad[index] + torch.sqrt(2*self.gamma)*gauss[index]
            P.data[index]+=g
            i+=1
            

    @torch.no_grad()
    def update_sparsity(self):
        ## Update all selected parameters structures
        i=0
        sparse_sum = 0.0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            Grad = P.grad.reshape(-1)[self.CoordSet[i]]
            Weights = P.data.reshape(-1)[self.CoordSet[i]]
            zz1 = self.a + 0.5*(self.rho_1 - self.rho_0)*Weights**2 -self.spleSize * -Grad*Weights
            prob = torch.sigmoid(-zz1)
            bern = torch.distributions.Binomial(1, prob).sample()
            params_struct_temp = self.params_struct[i].reshape(-1).to(self.device)
            params_struct_temp[self.CoordSet[i]] = bern
            self.params_struct[i] = params_struct_temp.reshape(self.params_struct[i].shape)
            sparse_sum += torch.sum(self.params_struct[i])
            i+=1
        print("sparsity: ", sparse_sum)
    
    
    def sparsify_model_params(self):
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            weights = P.data*self.params_struct[i].to(self.device)
            P.data = nn.parameter.Parameter(weights)
            i+=1