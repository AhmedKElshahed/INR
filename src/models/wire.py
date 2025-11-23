import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega0=30.0, sigma0=10.0, bias=True):
        super().__init__()
        self.omega_0 = omega0
        self.sigma_0 = sigma0
        self.is_first = is_first
        
        # We define separate Real and Imaginary parts for the weights.
        # This allows us to perform complex multiplication using standard float32 tensors,
        # which prevents DataParallel crashes and Dtype errors.
        
        # Frequency Term weights (Complex)
        self.freq_real = nn.Linear(in_features, out_features, bias=bias)
        self.freq_imag = nn.Linear(in_features, out_features, bias=bias)
        
        # Scale Term weights (Complex)
        self.scale_real = nn.Linear(in_features, out_features, bias=bias)
        self.scale_imag = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights(in_features)

    def init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                b = 1.0 / math.sqrt(in_features)
                val_range = (-b, b)
            else:
                s = 0.1 / math.sqrt(in_features)
                val_range = (-s, s)

            for layer in [self.freq_real, self.freq_imag, self.scale_real, self.scale_imag]:
                layer.weight.uniform_(*val_range)
                if layer.bias is not None:
                    layer.bias.zero_()

    def complex_linear(self, x, linear_r, linear_i):
        """
        Performs (x_real + i*x_imag) * (w_real + i*w_imag) + (b_real + i*b_imag)
        using only real arithmetic.
        """
        if torch.is_complex(x):
            xr, xi = x.real, x.imag
            # (xr + ixi)(wr + iwi) = (xr*wr - xi*wi) + i(xr*wi + xi*wr)
            
            # We use the layers directly. 
            # Note: linear_r(z) computes z*wr + br. 
            # To get pure matrix multiplication z*wr without adding bias twice, we use F.linear for the second parts.
            
            # Real part: xr*wr - xi*wi + br
            out_r = linear_r(xr) - F.linear(xi, linear_i.weight, None)
            
            # Imag part: xr*wi + xi*wr + bi
            out_i = linear_i(xr) + F.linear(xi, linear_r.weight, None)
        else:
            # Input is Real (first layer)
            # x * (wr + iwi) + (br + ibi)
            out_r = linear_r(x)
            out_i = linear_i(x)
            
        return out_r, out_i

    def forward(self, x):
        # 1. Flatten if DataParallel added extra dimensions
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        # 2. Apply Linear layers (Complex Multiplication)
        freq_r, freq_i = self.complex_linear(x, self.freq_real, self.freq_imag)
        scale_r, scale_i = self.complex_linear(x, self.scale_real, self.scale_imag)

        # 3. Gabor Nonlinearity
        # Frequency term: e^{i * omega * freq}
        # We use the Real part of the frequency projection to drive the oscillation (standard Gabor)
        # or the full complex magnitude? The paper implies complex sinusoid.
        # Let's use the full complex projection for the phase.
        
        # Phase = omega0 * (freq_r + i*freq_i)
        # e^{i * Phase} = e^{i * omega0 * freq_r - omega0 * freq_i}
        # This can explode if freq_i is large. 
        # Standard WIRE implementation usually takes just the Real part for oscillation.
        # Let's stick to: e^{i * omega * freq_r}
        
        phase = self.omega_0 * freq_r
        # Create complex sinusoid: cos(phase) + i*sin(phase)
        freq_term = torch.polar(torch.ones_like(phase), phase)

        # Envelope term: Gaussian
        # exp( -sigma^2 * |scale|^2 )
        # |scale|^2 = scale_r^2 + scale_i^2
        mag_sq = scale_r.square() + scale_i.square()
        gauss_term = torch.exp(-(self.sigma_0 ** 2) * mag_sq.clamp(max=20.0))
        
        # Result is complex
        return freq_term * gauss_term

class Wire(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4,
                 out_features=3, first_omega_0=30.0, hidden_omega_0=10.0, sigma0=10.0):
        super().__init__()
        
        layers = []
        layers.append(ComplexGaborLayer(in_features, hidden_features, is_first=True,
                                        omega0=first_omega_0, sigma0=sigma0))
        
        for _ in range(hidden_layers):
            layers.append(ComplexGaborLayer(hidden_features, hidden_features, is_first=False,
                                            omega0=hidden_omega_0, sigma0=sigma0))
            
        self.layers = nn.ModuleList(layers)
        
        # Final layer: Complex -> Real
        # We map hidden_features (complex) to out_features (real)
        # Output = Real( x * w ) + b
        self.final_linear_r = nn.Linear(hidden_features, out_features)
        self.final_linear_i = nn.Linear(hidden_features, out_features)
        
        with torch.no_grad():
            s = 0.1 / math.sqrt(hidden_features)
            self.final_linear_r.weight.uniform_(-s, s)
            self.final_linear_i.weight.uniform_(-s, s)
            self.final_linear_r.bias.zero_()
            # No bias for imag part needed since we discard it

    def forward(self, x):
        # Pass through Gabor Layers
        for l in self.layers:
            x = l(x)
        
        # Final Projection (Complex input x -> Real output)
        # x = xr + i*xi
        # w = wr + i*wi
        # Real(x*w) = xr*wr - xi*wi
        
        xr, xi = x.real, x.imag
        out = self.final_linear_r(xr) - F.linear(xi, self.final_linear_i.weight, None)
        
        return out
