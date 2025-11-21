import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset
import gc

class OccupancyDataset(Dataset):
    def __init__(self, mesh_path, num_samples=100000, on_surface_ratio=0.5):
        print(f"Loading mesh: {mesh_path}...")
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # --- STEP 1: Simplify Mesh (Crucial for RAM) ---
        # Reduce to 10,000 faces max. This keeps the shape but kills RAM usage.
        if len(mesh.faces) > 10000:
            print(f"-> Decimating mesh from {len(mesh.faces)} faces to ~10,000...")
            mesh = mesh.simplify_quadratic_decimation(10000)
            
        # 2. Normalize Mesh to unit sphere [-1, 1]
        vertices = mesh.vertices
        center = (vertices.max(0) + vertices.min(0)) / 2
        scale = 1.8 / (vertices.max(0) - vertices.min(0)).max()
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
        
        # 3. Sample Points
        print(f"Generating {num_samples} points...")
        
        # A. Uniform
        n_uniform = int(num_samples * (1 - ratio))
        points_uniform = np.random.rand(n_uniform, 3) * 2 - 1
        
        # B. Surface
        n_surface = num_samples - n_uniform
        points_surface, _ = trimesh.sample.sample_surface(mesh, n_surface)
        points_surface += np.random.normal(0, 0.01, points_surface.shape)
        
        all_points = np.concatenate([points_uniform, points_surface], axis=0)
        
        # C. Occupancy Check (Batched)
        print("-> Ray tracing (this might take 30s)...")
        occupancy = np.zeros(len(all_points), dtype=np.float32)
        chunk_size = 50000
        
        for i in range(0, len(all_points), chunk_size):
            end = min(i + chunk_size, len(all_points))
            # .contains is memory hungry, but we simplified the mesh so it's safe now
            occupancy[i:end] = mesh.contains(all_points[i:end]).astype(np.float32)
            
        # --- STEP 2: DELETE THE MESH (Crucial for DataLoader) ---
        # We don't need the geometry anymore, just the points.
        del mesh
        gc.collect()
        print("-> Mesh geometry deleted from memory.")

        # Convert to Torch
        self.points = torch.from_numpy(all_points).float()
        self.occupancies = torch.from_numpy(occupancy).float().unsqueeze(1)
        
        print(f"Dataset ready: {len(self.points)} points.")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.occupancies[idx]
