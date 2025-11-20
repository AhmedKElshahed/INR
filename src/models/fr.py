import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FRLinearPaper(nn.Module):
    def __init__(self, in_features, out_features, F=64, bias=True, activation=None, include_dc=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.F = int(F)
        self.include_dc = include_dc

        n = torch.arange(in_features, dtype=torch.float32)
        z = 2.0 * math.pi * n / float(in_features)

        rows = []
        if include_dc:
            rows.append(torch.ones_like(z))

        for k in range(1, self.F + 1):
            kz = k * z
            rows.append(torch.cos(kz))
            rows.append(torch.sin(kz))

        B = torch.stack(rows, dim=0)
        B = B / (B.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8))

        self.M = B.shape[0]
        self.register_buffer("B", B)

        self.Lambda = nn.Parameter(torch.empty(out_features, self.M))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        with torch.no_grad():
            nn.init.xavier_uniform_(self.Lambda)
            if self.bias is not None:
                self.bias.zero_()

        self.activation = activation or (lambda x: x)

    def forward(self, x):
        feat = x @ self.B.t()
        out = F.linear(feat, self.Lambda, self.bias)
        return self.activation(out)

class FRPaper(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4, out_features=3, F=64, include_dc=True):
        super().__init__()
        layers = [FRLinearPaper(in_features, hidden_features, F=F, include_dc=include_dc), nn.ReLU(inplace=True)]
        for _ in range(hidden_layers - 1):
            layers += [FRLinearPaper(hidden_features, hidden_features, F=F, include_dc=include_dc), nn.ReLU(inplace=True)]
        layers += [FRLinearPaper(hidden_features, out_features, F=F, include_dc=include_dc)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
