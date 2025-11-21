import torch
import numpy as np
from torch.utils.data import Dataset
import os

class OccupancyDataset(Dataset):
    def __init__(self, mesh_path, num_samples=None, on_surface_ratio=None):
        # Arguments are ignored because we load pre-processed data
        data_path = "dragon_dataset.npz"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find {data_path}. Please run generate_data.py first!")
            
        print(f"Loading pre-processed data from {data_path}...")
        data = np.load(data_path)
        
        self.points = torch.from_numpy(data['points'])
        # Convert uint8/bool to float [N, 1]
        self.occupancies = torch.from_numpy(data['occupancies']).float().unsqueeze(1)
        
        print(f"Dataset loaded successfully: {len(self.points)} points.")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.occupancies[idx]
