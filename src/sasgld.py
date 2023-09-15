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

class sasgldSampler(object):
    def __init__(self, device, config, model):
        self.device = device
        self.model = model
        self.lr = torch.from_numpy(np.array(config["sampler"]["learning_rate"])).to(dtype=torch.float32).to(self.device)
        self.gamma = torch.from_numpy(np.array(config["training"]["gamma"])).to(dtype=torch.float32).to(self.device)
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
                self.J_vec[i] = torch.ceil(self.update_rate*torch.prod(P))
                self.CoordSet[i] = torch.zeros(self.J_vec[i].numel())
            i+=1

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
            self.CoordSet[i] = np.random.choice(a=P.numel(), size = self.J_vec[i], replace = False)
            i+=1

    def update_layer_param(self, P, struct):
        gauss = torch.distributions.Normal(torch.zeros_like(P.data), torch.ones_like(P.data))
        print(gauss)
        index = struct==1
        P[struct==0].data = torch.sqrt(1/self.gamma)*gauss[struct==0]
        g = - self.lr *(self.spleSize*P.grad[index] + self.rho_1*P[index]) + np.sqrt(2*self.lr)*gauss[index]
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
        weight = self.a + 0.5*(self.rho_1 - self.rho_0)*P[CoordSet]**2 - P.grad[CoordSet] *P[CoordSet]
        prob = expit(-weight)
        struct[CoordSet] = torch.rand(CoordSet.numel()) <= prob

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
            P.data = P*self.param_struct[i]
            i+=1
        return self.model