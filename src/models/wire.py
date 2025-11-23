import torch
import torch.nn as nn
import math

class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega0=30.0, sigma0=10.0, bias=True):
        super().__init__()
        self.omega_0 = omega0
        self.sigma_0 = sigma0
        
        # We use real-valued implementations to avoid complex-tensor issues with DataParallel
        self.is_first = is_first
        
        # Linear layer for the frequency term
        self.linear_real = nn.Linear(in_features, out_features, bias=bias)
        self.linear_imag = nn.Linear(in_features, out_features, bias=bias)
        
        # Linear layer for the scale term
        self.linear_ortho_real = nn.Linear(in_features, out_features, bias=bias)
        self.linear_ortho_imag = nn.Linear(in_features, out_features, bias=bias)
        
        with torch.no_grad():
            if is_first:
                b = 1.0 / math.sqrt(in_features)
                self.linear_real.weight.uniform_(-b, b)
                self.linear_imag.weight.uniform_(-b, b)
                self.linear_ortho_real.weight.uniform_(-b, b)
                self.linear_ortho_imag.weight.uniform_(-b, b)
            else:
                s = 0.1 / math.sqrt(in_features)
                self.linear_real.weight.uniform_(-s, s)
                self.linear_imag.weight.uniform_(-s, s)
                self.linear_ortho_real.weight.uniform_(-s, s)
                self.linear_ortho_imag.weight.uniform_(-s, s)
                
            if bias:
                self.linear_real.bias.zero_()
                self.linear_imag.bias.zero_()
                self.linear_ortho_real.bias.zero_()
                self.linear_ortho_imag.bias.zero_()

    def forward(self, x):
        # x might be real or complex. We treat it as real [B, in_feat]
        # DataParallel might add an extra dimension [B, 1, in_feat] -> Fix it
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        # Frequency term (Complex linear)
        lin_r = self.linear_real(x)
        # We ignore imag part of input for simple 3D coords, but strictly:
        # (a+bi)(c+di) = (ac-bd) + i(ad+bc). 
        # Since input coords are real, imag input is 0. So just real weights matter.
        
        # Scale term
        ortho_r = self.linear_ortho_real(x)

        # Gabor Wavelet logic
        # Unit-magnitude complex sinusoid: e^{i * omega * lin_r}
        phase = self.omega_0 * lin_r
        freq_term = torch.polar(torch.ones_like(phase), phase)

        # Gaussian envelope
        gauss_arg = (lin_r.square() + ortho_r.square()).clamp(max=20.0)
        gauss_term = torch.exp(-(self.sigma_0 ** 2) * gauss_arg)

        return freq_term * gauss_term

class Wire(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4,
                 out_features=3, first_omega_0=30.0, hidden_omega_0=10.0, sigma0=10.0):
        super().__init__()
        
        # Build layers
        layers = []
        layers.append(ComplexGaborLayer(in_features, hidden_features, is_first=True,
                                        omega0=first_omega_0, sigma0=sigma0))
        
        for _ in range(hidden_layers):
            layers.append(ComplexGaborLayer(hidden_features, hidden_features, is_first=False,
                                            omega0=hidden_omega_0, sigma0=sigma0))
            
        self.layers = nn.ModuleList(layers)
        
        # Final layer is complex -> real
        # We implement it manually to be safe
        self.final_linear_real = nn.Linear(hidden_features, out_features)
        self.final_linear_imag = nn.Linear(hidden_features, out_features)
        
        with torch.no_grad():
            s = 0.1 / math.sqrt(hidden_features)
            self.final_linear_real.weight.uniform_(-s, s)
            self.final_linear_imag.weight.uniform_(-s, s)
            self.final_linear_real.bias.zero_()
            self.final_linear_imag.bias.zero_()

    def forward(self, x):
        # Forward pass
        for l in self.layers:
            x = l(x)
        
        # Final projection (Complex -> Real part only)
        # x is complex here.
        # result = x * weight + bias
        # Real part = (x.real * w.real - x.imag * w.imag) + bias
        out = (x.real @ self.final_linear_real.weight.t() - 
               x.imag @ self.final_linear_imag.weight.t()) + self.final_linear_real.bias
               
        return out
