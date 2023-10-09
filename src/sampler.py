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

class OptimizerTemplate:

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        ## Set gradients of all parameters to zero
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_() # For second-order optimizers important
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        ## Apply update step to all parameters
        for p in self.params:
            if p.grad is None: # We skip parameters without any gradients
                continue
            self.update_param(p)

    def update_param(self, p):
        # To be implemented in optimizer-specific classes
        raise NotImplementedError


class SGD(OptimizerTemplate):

    def __init__(self, params, lr):
        super().__init__(params, lr)

    def update_param(self, p):
        update = -self.lr * p.grad
        p.add_(update) # In-place update => saves memory and does not create computation graph
        
class SGLD(OptimizerTemplate):

    def __init__(self, params, lr, spleSize):
        super().__init__(params, lr)
        self.spleSize = spleSize
    
    def update_param(self, p):
        gauss = torch.distributions.Normal(torch.zeros_like(p.data),torch.ones_like(p.data)).sample()
        update = -self.spleSize*self.lr * p.grad + np.sqrt(2*self.lr)*gauss
        p.add_(update) # In-place update => saves memory and does not create computation graph
        
class SASGLD(object):
    def __init__(self, device, config, model):
        self.device = device
        self.model = model
        #self.lr = torch.tensor(config["sampler"]["learning_rate"], dtype=torch.float32).to(self.device)
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
    

    def zero_grad(self, model):
        ## Set gradients of all parameters to zero
        for P in model.parameters():
            if P.grad is not None:
                P.grad.detach_() # For second-order optimizers important
                P.grad.zero_()
        return model        
        

    def select_CoordSet(self, model):
        i=0
        for P in model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            self.CoordSet[i] = np.random.choice(a=P.numel(), size = int(self.J_vec[i].item()), replace = False)
            i+=1


    @torch.no_grad()
    def update_params(self, model):
        ## Update all parameters
        i=0
        for P in model.parameters():
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
        return model    

    @torch.no_grad()
    def update_sparsity(self, model):
        ## Update all selected parameters structures
        i=0
        sparse_sum = 0.0
        for P in model.parameters():
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
        print("sparsity: ", (sparse_sum/self.total_par).item())
        return (sparse_sum/self.total_par).item()
    
    
    def sparsify_model_params(self, model):
        i=0
        for P in model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            weights = P.data*self.params_struct[i].to(self.device)
            P.data = nn.parameter.Parameter(weights)
            i+=1
        return model
    
    def reinit_sparsity(self, model):
        i = 0
        for P in model.parameters():
            if P.grad is None: # We skip parameters without any gradients
                i+=1
                continue
            gauss = torch.distributions.Normal(torch.zeros_like(P.data), torch.ones_like(P.data)).sample()
            P[self.params_struct[i]==0].data = gauss[self.params_struct[i]==0]
            self.params_struct[i]=torch.ones(size=self.params_struct[i].shape)
            i+=1
        return model