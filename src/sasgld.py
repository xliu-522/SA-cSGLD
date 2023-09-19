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
        self.rho_0 = config["sampler"]["rho_0"]
        self.rho_1 = config["sampler"]["rho_1"]
        self.spleSize = config["data"]["sample_size"]
        self.u = config["sampler"]["u"]
        self.total_par = config["model"]["total_par"]
        self.update_rate = config["sampler"]["update_rate"]

        # Initiate parameter structure
        self.params_struct, self.J_vec, self.CoordSet  = {}, {}, {}
        i = 0
        for P in self.model.parameters():
            if P.requires_grad:
                self.params_struct[i] = torch.ones(size=P.shape)
                self.J_vec[i] = torch.ceil(self.update_rate*torch.prod(torch.tensor(P.shape)))
                self.CoordSet[i] = torch.zeros(self.J_vec[i].numel())
            i+=1
        self.a = self.u * np.log(self.total_par) + 0.5*np.log(self.rho_0/self.rho_1)

    def zero_grad(self):
        ## Set gradients of all parameters to zero
        for P in self.model.parameters():
            if P.grad is not None:
                P.grad.detach_() # For second-order optimizers important
                P.grad.zero_()

    def select_CoordSet(self):
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            self.CoordSet[i] = np.random.choice(a=P.numel(), size = int(self.J_vec[i].item()), replace = False)
            i+=1

    def update_layer_param(self, P, struct):
        gauss_dist = torch.distributions.Normal(torch.zeros_like(P.data), torch.ones_like(P.data))
        gauss = gauss_dist.sample()
        L = torch.sum(struct==0)
        if L > 0:
            P[struct==0].data = torch.sqrt(1/self.rho_0)*gauss[struct==0]
        L = torch.sum(struct==1)
        if L > 0:
            index = struct==1
            g = self.gamma * (- self.rho_1*P.data[index] + self.spleSize*P.grad[index]) + torch.sqrt(2*self.gamma)*gauss[index]
            P[index].data +=g

    @torch.no_grad()
    def update_params(self):
        ## Update all parameters
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            self.update_layer_param(P, self.params_struct[i])
            i+=1
            
    def update_layer_sparsity(self, P, struct, CoordSet):
        Grad = P.grad.reshape(-1)[CoordSet]
        Weights = P.data.reshape(-1)[CoordSet]
        zz = self.a + 0.5*(self.rho_1 - self.rho_0)*Weights**2 - Grad *Weights
        prob = torch.sigmoid(-zz)
        struct.to(self.device).reshape(-1)[CoordSet] = torch.distributions.Binomial(1, prob).sample()

    @torch.no_grad()
    def update_sparsity(self):
        ## Update all selected parameters structures
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            self.update_layer_sparsity(P, self.params_struct[i], self.CoordSet[i])
            i+=1
    
    def sparsify_model_params(self):
        i=0
        for P in self.model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            P.data = P.data*self.params_struct[i].to(self.device)
            i+=1 
        return self.model