import torch
import torch.nn as nn
import math

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=3, 
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        
        self.net = []
        
        # 1. First Layer
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        # 2. Hidden Layers
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        # 3. Final Linear Layer
        final_linear = nn.Linear(hidden_features, out_features)
        
        # Initialize final layer (Weights scaled, Bias default)
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_features) / hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)
            # Note: We DO NOT zero the bias here, matching the reference experiment.
                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output
