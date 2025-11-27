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
        
        # Apply specific initialization
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for i, m in enumerate(self.modules()):
                if isinstance(m, nn.Linear):
                    # Standard initialization for weights
                    std = 1.0 / math.sqrt(m.weight.size(1))
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                    
                    if m.bias is not None:
                        # CRITICAL FIX:
                        # For the first layer, spread biases across the input scale
                        # so Gaussians are positioned all over the image, not just at 0.
                        if i == 1: # The first Linear layer is usually index 1 in modules()
                            nn.init.uniform_(m.bias, -self.input_scale, self.input_scale)
                        else:
                            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Scale inputs to drive high-frequency features
        return self.net(x * self.input_scale)
