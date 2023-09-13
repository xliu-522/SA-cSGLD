import torch
import random
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size = 11, stride = 4, padding =1),
            nn.ReLU(), nn.MaxPool2d(kernel_size = 3, stride =2),
            nn.LazyConv2d(256, kernel_size = 5, padding = 2),nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride =2),
            nn.LazyConv2d(384, kernel_size = 3, padding = 1),nn.ReLU(),
            nn.LazyConv2d(384, kernel_size = 3, padding = 1),nn.ReLU(),
            nn.LazyConv2d(256, kernel_size = 3, padding = 1),nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride =2), nn.Flatten(),
            nn.LazyLinear(4096),nn.ReLU(),
            nn.LazyLinear(4096),nn.ReLU(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, x):
        logits = self.net(x)
        return logits