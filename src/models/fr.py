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

        # ====================================================================
        # CRITICAL FIX: COORDINATE MAPPING
        # ====================================================================
        # For 2D Images and 3D Shapes, we MUST use Random Fourier Features (RFF).
        # The deterministic formula (n/in_features) creates a Rank-1 matrix
        # which blinds the model to details off the diagonal.
        
        if in_features <= 4: 
            # --- OPTION A: RANDOM SAMPLING (Required for 2D/3D) ---
            # Standard Gaussian initialization (Sigma=10.0 is standard for coords)
            # This ensures the model "looks" in all directions (isotropic).
            B = torch.randn(self.F, in_features) * 10.0
            
            # We register B as a buffer (saved with model, but not trained)
            self.register_buffer("B_random", B)
            self.use_random = True
            
            # Feature count: DC (1) + Sin(F) + Cos(F)
            self.M = (1 if include_dc else 0) + 2 * self.F
            
        else:
            # --- OPTION B: DETERMINISTIC (High-Dim Data Only) ---
            # This matches the formula you posted, but is only valid
            # if in_features is large (e.g., raw audio chunks).
            self.use_random = False
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
            # Normalize rows
            B = B / (B.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8))
            
            self.M = B.shape[0]
            self.register_buffer("B", B)

        # ====================================================================
        # LEARNABLE WEIGHTS (Spectral Mixing)
        # ====================================================================
        self.Lambda = nn.Parameter(torch.empty(out_features, self.M))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialization
        with torch.no_grad():
            nn.init.xavier_uniform_(self.Lambda)
            if self.bias is not None:
                self.bias.zero_()

        self.activation = activation or (lambda x: x)

    def forward(self, x):
        if self.use_random:
            # 1. Random Projection: [Batch, In] @ [In, F] -> [Batch, F]
            proj = x @ self.B_random.t()
            
            # 2. Create Sin/Cos Features: [Batch, 2*F]
            proj_cat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            
            # 3. Add DC term if needed
            if self.include_dc:
                ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
                feat = torch.cat([ones, proj_cat], dim=-1)
            else:
                feat = proj_cat
        else:
            # Deterministic Projection
            feat = x @ self.B.t()
            
        # 4. Learnable Mixing
        out = F.linear(feat, self.Lambda, self.bias)
        return self.activation(out)

class FRPaper(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=4, out_features=3, F=64, include_dc=True):
        super().__init__()
        
        # Input Layer
        layers = [FRLinearPaper(in_features, hidden_features, F=F, include_dc=include_dc), nn.ReLU(inplace=True)]
        
        # Hidden Layers
        for _ in range(hidden_layers - 1):
            layers += [FRLinearPaper(hidden_features, hidden_features, F=F, include_dc=include_dc), nn.ReLU(inplace=True)]
            
        # Output Layer
        layers += [FRLinearPaper(hidden_features, out_features, F=F, include_dc=include_dc)]
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
