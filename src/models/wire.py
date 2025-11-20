import torch
import torch.nn as nn
import math

class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega0=30.0, sigma0=10.0, bias=True):
        super().__init__()
        self.omega_0 = omega0
        self.sigma_0 = sigma0
        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.linear_ortho = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        with torch.no_grad():
            if is_first:
                b = 1.0 / math.sqrt(in_features)
                self.linear.weight.uniform_(-b, b)
                self.linear_ortho.weight.uniform_(-b, b)
                if bias:
                    self.linear.bias.zero_()
                    self.linear_ortho.bias.zero_()
            else:
                s = 0.1 / math.sqrt(in_features)
                self.linear.weight.real.uniform_(-s, s)
                self.linear.weight.imag.uniform_(-s, s)
                self.linear_ortho.weight.real.uniform_(-s, s)
                self.linear_ortho.weight.imag.uniform_(-s, s)
                if bias:
                    self.linear.bias.data.zero_()
                    self.linear_ortho.bias.data.zero_()

    def forward(self, x):
        lin  = self.linear(x)
        ortho = self.linear_ortho(x)
        lin_r   = lin.real if torch.is_complex(lin)   else lin
        ortho_r = ortho.real if torch.is_complex(ortho) else ortho
        phase = self.omega_0 * lin_r
        freq_term = torch.polar(torch.ones_like(phase), phase)
        gauss_arg = (lin_r.square() + ortho_r.square()).clamp(max=20.0)
        gauss_term = torch.exp(-(self.sigma_0 ** 2) * gauss_arg)
        return freq_term * gauss_term

class Wire(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4,
                 out_features=3, first_omega_0=30.0, hidden_omega_0=10.0, sigma0=10.0):
        super().__init__()
        layers = [ComplexGaborLayer(in_features, hidden_features, is_first=True,
                                    omega0=first_omega_0, sigma0=sigma0)]
        for _ in range(hidden_layers):
            layers.append(ComplexGaborLayer(hidden_features, hidden_features, is_first=False,
                                            omega0=hidden_omega_0, sigma0=sigma0))
        self.layers = nn.ModuleList(layers)
        self.final = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)
        with torch.no_grad():
            s = 0.1 / math.sqrt(hidden_features)
            self.final.weight.real.uniform_(-s, s); self.final.weight.imag.uniform_(-s, s)
            self.final.bias.data.zero_()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        out = self.final(x)
        return out.real
