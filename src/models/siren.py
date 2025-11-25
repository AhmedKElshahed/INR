import torch
import torch.nn as nn
import math

class SineLayer(nn.Module):
    """
    A Linear Layer followed by a Sine activation.
    
    Key difference from standard layers:
    1. The Sine activation frequency is scaled by omega_0.
    2. Weights are initialized specifically to keep signal magnitude constant 
       through the network (Paper Sec 3.2).
    """
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
                # First layer: Uniform(-1/fan_in, 1/fan_in)
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                # Hidden layers: Uniform(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)
                # This scaling factor is crucial for deep SIRENs
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
        
        # 1. First Layer (uses first_omega_0, usually 30 or 80 for images)
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        # 2. Hidden Layers (uses hidden_omega_0, usually 30)
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        # 3. Final Linear Layer
        # We use a Linear layer (not Sine) at the end for SR/Occupancy to output raw values (RGB or Logits)
        final_linear = nn.Linear(hidden_features, out_features)
        
        # Initialize final layer carefully
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_features) / hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)
            if final_linear.bias is not None:
                final_linear.bias.zero_()
                
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # Allows taking derivatives w.r.t input (needed for some PDE tasks, harmless here)
        # coords = coords.clone().detach().requires_grad_(True) 
        output = self.net(coords)
        return output
