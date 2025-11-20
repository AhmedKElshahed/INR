import torch
import torch.nn as nn

class _INCodeComposerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, a, w, phi, b):
        z = self.linear(x)
        return a * torch.sin(w * z + phi) + b

class INCode(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4,
                 out_features=3, scale=256, prior_dim=64):
        super().__init__()
        self.first = _INCodeComposerLayer(in_features, hidden_features)
        self.hiddens = nn.ModuleList([_INCodeComposerLayer(hidden_features, hidden_features) for _ in range(hidden_layers)])
        self.final = nn.Linear(hidden_features, out_features)
        self.prior = nn.Parameter(torch.randn(prior_dim))
        total_layers = 1 + hidden_layers
        self.harmonizer = nn.Sequential(
            nn.Linear(prior_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, total_layers * 4)
        )
        self.per_neuron = nn.ModuleList([nn.Linear(1, hidden_features, bias=False) for _ in range(total_layers)])
        with torch.no_grad():
            nn.init.normal_(self.final.weight, 0, 0.01)
            if self.final.bias is not None: self.final.bias.zero_()

    def _expand(self, layer_idx, base_params, C, device):
        a0, w0, p0, b0 = base_params
        inp = torch.ones(1, 1, device=device)
        expand = self.per_neuron[layer_idx](inp)
        a = torch.sigmoid(a0) * expand
        w = torch.exp(w0).clamp(0.1, 100.0) * torch.ones_like(expand)
        phi = p0 * torch.ones_like(expand)
        b = b0 * torch.ones_like(expand)
        return a, w, phi, b

    def forward(self, x):
        dev = x.device
        hvec = self.harmonizer(self.prior)
        L = len(self.hiddens) + 1
        hvec = hvec.view(L, 4)
        a,w,phi,b = self._expand(0, hvec[0], self.first.linear.out_features, dev)
        h = self.first(x, a, w, phi, b)
        for i, layer in enumerate(self.hiddens, start=1):
            a,w,phi,b = self._expand(i, hvec[i], layer.linear.out_features, dev)
            h = layer(h, a, w, phi, b)
        return self.final(h)
