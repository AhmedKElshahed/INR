import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset

class OccupancyDataset(Dataset):
    def __init__(self, mesh_path, num_samples=100000, on_surface_ratio=0.5):
        """
        mesh_path: Path to .obj/.off/.ply file
        num_samples: Total points to sample for training
        on_surface_ratio: How many points should be near surface vs random uniform
        """
        print(f"Loading mesh: {mesh_path}...")
        # Force mesh loading to ensure we have geometry
        self.mesh = trimesh.load(mesh_path, force='mesh')
        
        # 1. Normalize Mesh to unit sphere [-1, 1]
        vertices = self.mesh.vertices
        center = (vertices.max(0) + vertices.min(0)) / 2
        scale = 1.8 / (vertices.max(0) - vertices.min(0)).max() # 1.8 ensures margin
        self.mesh.apply_translation(-center)
        self.mesh.apply_scale(scale)
        
        print(f"Generating {num_samples} training points...")
        self.points, self.occupancies = self._sample_points(num_samples, on_surface_ratio)
        
        # Convert to Torch
        self.points = torch.from_numpy(self.points).float()
        self.occupancies = torch.from_numpy(self.occupancies).float().unsqueeze(1) # [N, 1]
        
        print(f"Dataset ready: {len(self.points)} points.")

    def _sample_points(self, num_samples, ratio):
        # A. Uniform sampling within [-1, 1]
        n_uniform = int(num_samples * (1 - ratio))
        points_uniform = np.random.rand(n_uniform, 3) * 2 - 1
        
        # B. Near-surface sampling (critical for detail)
        n_surface = num_samples - n_uniform
        points_surface, _ = trimesh.sample.sample_surface(self.mesh, n_surface)
        # Add slight noise to surface points to get inside/outside
        points_surface += np.random.normal(0, 0.01, points_surface.shape)
        
        # Combine
        all_points = np.concatenate([points_uniform, points_surface], axis=0)
        
        # C. Check occupancy (Batched to prevent CPU Freeze)
        print("-> Running occupancy check (Ray Tracing)...")
        occupancy = np.zeros(len(all_points), dtype=np.float32)
        
        # Process in chunks of 50k to keep RAM low
        chunk_size = 50000
        for i in range(0, len(all_points), chunk_size):
            end = min(i + chunk_size, len(all_points))
            batch_points = all_points[i:end]
            
            # contains returns boolean, convert to float 0.0/1.0
            occupancy[i:end] = self.mesh.contains(batch_points).astype(np.float32)
            
            # Simple progress indicator
            if i % (chunk_size * 4) == 0:
                print(f"   Processed {i}/{len(all_points)} points...")
        
        return all_points, occupancy

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.occupancies[idx]
