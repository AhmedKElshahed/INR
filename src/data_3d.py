import torch
import numpy as np
from torch.utils.data import Dataset
import os

class OccupancyDataset(Dataset):
    def __init__(self, dataset_path):
        # Logic updated: We now strictly trust the path passed to us
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could not find {dataset_path}. Please run generate_data.py first!")
            
        print(f"Loading pre-processed data from {dataset_path}...")
        data = np.load(dataset_path)
        
        self.points = torch.from_numpy(data['points'])
        self.occupancies = torch.from_numpy(data['occupancies']).float().unsqueeze(1)
        
        print(f"Dataset loaded successfully: {len(self.points)} points.")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.occupancies[idx]
