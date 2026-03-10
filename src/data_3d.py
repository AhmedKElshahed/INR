import torch
import numpy as np
from torch.utils.data import Dataset
import os

class OccupancyDataset(Dataset):
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could not find {dataset_path}. Please run generate_data.py first!")
            
        print(f"Loading pre-processed data from {dataset_path}...")
        data = np.load(dataset_path)
        
        # Load points
        self.points = torch.from_numpy(data['points']).float()

        min_vals = self.points.min(dim=0)[0]
        max_vals = self.points.max(dim=0)[0]
        center = (min_vals + max_vals) / 2
        self.points = self.points - center
        
        # 2. Scale to [-1, 1] (preserving aspect ratio)
        max_dist = self.points.abs().max()
        if max_dist > 0:
            self.points = self.points / max_dist
            
        print(f"-> Data Normalized. Range: [{self.points.min():.3f}, {self.points.max():.3f}]")
        # Store normalization parameters so the training script can put the GT mesh
        # into the same coordinate space as the predicted (marching-cubes) mesh.
        self.norm_center = center.numpy()
        self.norm_max_dist = max_dist.item()
        # ---------------------------------------------

        self.occupancies = torch.from_numpy(data['occupancies']).float().unsqueeze(1)

        print(f"Dataset loaded successfully: {len(self.points)} points.")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.occupancies[idx]
