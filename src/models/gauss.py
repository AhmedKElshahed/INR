import torch
import torch.nn as nn
import math

class GaussianActivation(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        return torch.exp(-(x ** 2) / (2 * self.sigma ** 2))

class GaussianMLP(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4, out_features=3, 
                 input_scale=256.0, sigma=1.0):
        super().__init__()
        self.input_scale = input_scale
        
        layers = []
        # First layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(GaussianActivation(sigma))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(GaussianActivation(sigma))
            
        # Final layer (Linear output)
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        
        # Initialization (Crucial for Gaussian MLPs)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # Use Xavier/Kaiming-like init but scaled for Gaussian stability
                    std = math.sqrt(2.0 / hidden_features)
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        # CRITICAL FIX: Scale inputs so the Gaussian functions cover the domain
        return self.net(x * self.input_scale)
