import torch
import torch.nn as nn
import math

class GaussianActivation(nn.Module):
    def __init__(self, sigma_init=1.0, trainable=True):
        super().__init__()
        # Make Sigma a learnable parameter per layer
        if trainable:
            self.sigma = nn.Parameter(torch.tensor(sigma_init))
        else:
            self.register_buffer("sigma", torch.tensor(sigma_init))
    
    def forward(self, x):
        return torch.exp(-(x ** 2) / (2 * self.sigma ** 2))

class GaussianMLP(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4, out_features=3, 
                 input_scale=32.0, sigma=1.0): # Note: Default scale lowered to 32.0
        super().__init__()
        self.input_scale = input_scale
        
        layers = []
        # First layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(GaussianActivation(sigma, trainable=True))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(GaussianActivation(sigma, trainable=True))
            
        # Final layer (Linear output)
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for i, m in enumerate(self.net):
                if isinstance(m, nn.Linear):
                    # Standard Xavier/Kaiming init
                    std = 1.0 / math.sqrt(m.weight.size(1))
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                    nn.init.zeros_(m.bias)

            # CRITICAL: Spread the first layer to cover the image
            # Access first linear layer safely
            first_linear = self.net[0] 
            nn.init.uniform_(first_linear.bias, -self.input_scale, self.input_scale)
            # Scale weights too so they project into the right range
            nn.init.uniform_(first_linear.weight, -self.input_scale, self.input_scale)

    def forward(self, x):
        # We removed the manual scaling here because we baked it into the initialization.
        # This is more numerically stable.
        return self.net(x)
