import torch
import torch.nn as nn
import math


class GaussianActivation(nn.Module):
    def __init__(self, sigma_init=1.0, trainable=True):
        super().__init__()
        if trainable:
            self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))
        else:
            self.register_buffer("sigma", torch.tensor(float(sigma_init)))

    def forward(self, x):
        return torch.exp(-(x ** 2) / (2 * self.sigma ** 2))


class GaussianMLP(nn.Module):
    """
    Gaussian activation MLP for Implicit Neural Representations.

    Fix: The first layer pre-activations have std ≈ input_scale / sqrt(in_features).
    We set first_sigma to match this scale so the Gaussian activation is not
    permanently dead at init (which was the original bug with sigma=1 and large weights).
    """
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4, out_features=3,
                 input_scale=32.0, sigma=1.0):
        super().__init__()
        self.input_scale = input_scale

        # Effective sigma for first layer:
        # With Kaiming init (std=1/sqrt(in)) and input scaled by input_scale,
        # the pre-activation std ≈ input_scale / sqrt(in_features).
        # Setting first_sigma to this value keeps exp(-z^2/2σ^2) ≈ 0.6 at 1-std inputs.
        first_sigma = float(input_scale) / math.sqrt(float(in_features))

        # Separate linears and activations so the first layer can scale its input
        linears = [nn.Linear(in_features, hidden_features)]
        activations = [GaussianActivation(first_sigma, trainable=True)]

        for _ in range(hidden_layers):
            linears.append(nn.Linear(hidden_features, hidden_features))
            activations.append(GaussianActivation(sigma, trainable=True))

        self.linears = nn.ModuleList(linears)
        self.activations = nn.ModuleList(activations)
        self.final = nn.Linear(hidden_features, out_features)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for lin in self.linears:
                nn.init.kaiming_normal_(lin.weight, nonlinearity='linear')
                nn.init.zeros_(lin.bias)
            # Small output init for stable early-training BCE loss
            nn.init.normal_(self.final.weight, 0.0, 0.01)
            nn.init.zeros_(self.final.bias)

    def forward(self, x):
        # Scale coordinates for the first layer only
        h = self.activations[0](self.linears[0](x * self.input_scale))
        for lin, act in zip(self.linears[1:], self.activations[1:]):
            h = act(lin(h))
        return self.final(h)
