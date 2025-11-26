import torch
import torch.nn as nn
import math

# --- BASE ---
class MFNBase(nn.Module):
    def __init__(self, hidden_size, out_size, n_layers, weight_scale=1.0):
        super().__init__()
        self.mixers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(max(0, n_layers))])
        self.output = nn.Linear(hidden_size, out_size)
        with torch.no_grad():
            for m in self.mixers:
                m.weight.uniform_(-math.sqrt(weight_scale/hidden_size), math.sqrt(weight_scale/hidden_size))
                if m.bias is not None: m.bias.zero_()

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.mixers[i-1](out)
        return self.output(out)

# --- GABOR MFN ---
class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=6.0, beta=1.0, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.alpha = alpha
        self.beta = beta
        with torch.no_grad():
            nn.init.normal_(self.linear.weight, 0, math.sqrt(2.0 / in_features))
            if bias and self.linear.bias is not None:
                self.linear.bias.uniform_(-math.pi, math.pi)

    def forward(self, x):
        z = self.linear(x)
        return torch.exp(-self.beta * z * z) * torch.cos(self.alpha * z)

class GaborMFN(MFNBase):
    def __init__(self, in_size=2, hidden_size=256, out_size=3, n_layers=4,
                 input_scale=256., alpha=6.0, beta=1.0, weight_scale=1.0):
        super().__init__(hidden_size, out_size, n_layers, weight_scale)
        per_layer_scale = input_scale / math.sqrt(n_layers + 1)
        per_layer_alpha = alpha / (n_layers + 1)
        effective_beta = beta / (per_layer_scale ** 2)
        self.filters = nn.ModuleList([
            GaborLayer(in_size, hidden_size, alpha=per_layer_alpha, beta=effective_beta)
            for _ in range(n_layers + 1)
        ])
        with torch.no_grad():
            for f in self.filters:
                f.linear.weight.mul_(per_layer_scale)

# --- FOURIER MFN ---
class _FourierPE(nn.Module):
    def __init__(self, in_features=2, num_frequencies=256, sigma=10.0):
        super().__init__()
        B = torch.randn(in_features, num_frequencies) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2*math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class FourierMFN(nn.Module):
    def __init__(self, in_size=2, hidden_size=256, out_size=3, n_layers=4,
                 gamma=256.0, sigma=10.0):
        super().__init__()
        m = int(gamma) if gamma >= 1 else 256
        self.pe = _FourierPE(in_features=in_size, num_frequencies=m, sigma=sigma)
        fdim = 2 * m
        layers = [nn.Linear(fdim, hidden_size), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_size, out_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.pe(x))

# --- GAUSSIAN MFN ---
class GaussianLayer(nn.Module):
    def __init__(self, in_features, out_features, sigma=1.5, input_scale=256., bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.sigma = sigma
        self.input_scale = input_scale
        with torch.no_grad():
            nn.init.normal_(self.linear.weight, 0, math.sqrt(2.0 / in_features))
            if bias and self.linear.bias is not None:
                self.linear.bias.uniform_(-math.pi, math.pi)

    def forward(self, x):
        z = self.linear(x * self.input_scale)
        return torch.exp(- (z / self.sigma) ** 2)

class GaussianMFN(MFNBase):
    def __init__(self, in_size=2, hidden_size=256, out_size=3, n_layers=4,
                 input_scale=256., sigma=1.5, weight_scale=1.0):
        super().__init__(hidden_size, out_size, n_layers, weight_scale)
        per_layer_scale = input_scale / math.sqrt(n_layers + 1)
        self.filters = nn.ModuleList([
            GaussianLayer(in_size, hidden_size, sigma=sigma, input_scale=per_layer_scale)
            for _ in range(n_layers + 1)
        ])

# --- CORRECT GAUSSIAN MLP (Matches Paper) ---
class GaussianActivation(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        return torch.exp(-(x ** 2) / (2 * self.sigma ** 2))

class GaussianMLP(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4, out_features=3, sigma=1.0):
        super().__init__()
        
        # First layer
        layers = [nn.Linear(in_features, hidden_features), GaussianActivation(sigma)]
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(GaussianActivation(sigma))
            
        # Final layer (Linear, no activation)
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        
        # Initialization (Standard for Gaussian MLPs)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

