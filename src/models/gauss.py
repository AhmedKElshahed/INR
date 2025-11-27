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
            # 1. Standard Initialization for ALL layers first
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # Standard initialization for weights (1 / sqrt(fan_in))
                    std = 1.0 / math.sqrt(m.weight.size(1))
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                    nn.init.zeros_(m.bias)

            # 2. CRITICAL FIX: Explicitly initialize the First Layer Bias
            # We access the first layer directly using self.net[0]
            # This ensures Gaussians are spread across the entire input domain [-scale, scale]
            # instead of being clumped at 0.
            first_layer = self.net[0]
            nn.init.uniform_(first_layer.bias, -self.input_scale, self.input_scale)

    def forward(self, x):
        # Scale inputs to drive high-frequency features
        return self.net(x * self.input_scale)
