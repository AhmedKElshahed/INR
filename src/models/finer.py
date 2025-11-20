import torch
import torch.nn as nn
import math

class FINERLayer(nn.Module):
    def __init__(self, in_features, out_features, init_log_freq=0.0, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.log_freq = nn.Parameter(torch.full((out_features,), init_log_freq))
        self.phase = nn.Parameter(torch.zeros(out_features))
        with torch.no_grad():
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='linear')
            if bias and self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x):
        z = self.linear(x)
        w = torch.exp(self.log_freq).unsqueeze(0)
        p = self.phase.unsqueeze(0)
        return torch.sin(w * z + p)

class FINER(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4,
                 out_features=3, frequency_bands=6):
        super().__init__()
        layers = [FINERLayer(in_features, hidden_features, init_log_freq=torch.log(torch.tensor(float(max(1, frequency_bands))))) ]
        for _ in range(hidden_layers):
            layers.append(FINERLayer(hidden_features, hidden_features, init_log_freq=0.0))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, 0, 0.01)
            if self.net[-1].bias is not None: self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)
