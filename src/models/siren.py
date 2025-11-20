import torch
import torch.nn as nn
import math

def _siren_init_(weight, in_features, omega_0):
    bound = math.sqrt(6 / in_features) / omega_0
    with torch.no_grad():
        weight.uniform_(-bound, bound)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                _siren_init_(self.linear.weight, self.in_features, self.omega_0)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class Siren(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4,
                 out_features=3, first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        final = nn.Linear(hidden_features, out_features)
        _siren_init_(final.weight, hidden_features, hidden_omega_0)
        with torch.no_grad():
            if final.bias is not None:
                final.bias.zero_()
        self.net = nn.Sequential(*layers, final)

    def forward(self, x):
        return self.net(x)
