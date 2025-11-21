import os
import time
import torch
import torch.nn as nn
import numpy as np
import csv
from torch.utils.data import DataLoader
import skimage.measure
import trimesh

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models import create_model
from src.data_3d import OccupancyDataset
from src.config import BEST_CONFIGS

# ============================================================================
# 3D UTILS
# ============================================================================

def calculate_iou(pred, target, threshold=0.5):
    """Intersection over Union for binary occupancy"""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()

def extract_mesh(model, resolution=128, threshold=0.5, device='cuda'):
    """Runs Marching Cubes to extract the mesh from the INR"""
    model.eval()
    
    # Create grid
    grid_points = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
    points_t = torch.from_numpy(points).float().to(device)
    
    # Query model in chunks
    outputs = []
    # Increase chunk size for multi-GPU (2x T4 can handle 100k+ easily)
    chunk_size = 100000 
    
    with torch.no_grad():
        for i in range(0, len(points_t), chunk_size):
            batch = points_t[i:i+chunk_size]
            out = model(batch)
            outputs.append(out.cpu())
            
    outputs = torch.cat(outputs, dim=0).numpy().reshape(resolution, resolution, resolution)
    
    # Marching Cubes
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(outputs, level=threshold)
        # Normalize verts back to [-1, 1]
        verts = verts / (resolution - 1) * 2 - 1
        return trimesh.Trimesh(vertices=verts, faces=faces)
    except ValueError:
        print("Warning: No surface found at this threshold.")
        return None

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_occupancy(model_name, mesh_path, epochs=20, batch_size=16384, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on 3D Occupancy")
    print(f"GPUs Available: {gpu_count} ({torch.cuda.get_device_name(0)})")
    print(f"{'='*60}")

    # 1. Config & Model
    cfg = BEST_CONFIGS[model_name].copy()
    hidden_layers = cfg.pop("hidden_layers", 4)
    model = create_model(model_name, in_features=3, out_features=1, hidden_layers=hidden_layers, **cfg)
    
    # [OPTIMIZATION] Multi-GPU Support
    model = model.to(device)
    if gpu_count > 1:
        print(f"-> Activating DataParallel on {gpu_count} GPUs")
        model = nn.DataParallel(model)

    # 2. Data
    # We sample 1 million points (Good balance for speed/quality on Kaggle)
    print("-> Generating dataset (this should take <30s if rtree is installed)...")
    dataset = OccupancyDataset(mesh_path, num_samples=1000000, on_surface_ratio=0.5)
    
    # [OPTIMIZATION] Faster Data Loading
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,      # Use 2 CPU cores to fetch data
        pin_memory=True,    # Faster CPU->GPU transfer
        persistent_workers=True # Keep workers alive between epochs
    )

    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. Train
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_iou = 0
        steps = 0
        
        for p, t in dataloader:
            p, t = p.to(device), t.to(device)
            
            optimizer.zero_grad()
            preds = model(p)
            loss = criterion(preds, t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                total_iou += calculate_iou(preds, t)
            steps += 1
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/steps:.5f} | Train IoU: {total_iou/steps:.4f}")

    duration = time.time() - start_time

    # 5. Extract and Save Mesh
    os.makedirs("outputs_3d", exist_ok=True)
    print("-> Extracting mesh via Marching Cubes (resolution 128)...")
    
    # Resolution 128 is fast and looks decent. 
    # Go to 256 if you have time, but 128 prevents memory OOM on T4.
    rec_mesh = extract_mesh(model, resolution=128, device=device)
    
    if rec_mesh:
        out_path = f"outputs_3d/{model_name}_dragon.obj"
        rec_mesh.export(out_path)
        print(f"-> Saved mesh to {out_path}")
    
    return duration

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    MESH_PATH = "dragon.obj" 
    
    if not os.path.exists(MESH_PATH):
        print(f"Error: Could not find {MESH_PATH}. Please download a mesh first.")
        exit()

    # Check for fast ray tracing
    try:
        import rtree
        print("Success: 'rtree' is installed. Data generation will be fast.")
    except ImportError:
        print("WARNING: 'rtree' not found. Data generation will be SLOW (CPU 100%).")
        print("Run: !apt-get install -y libspatialindex-dev && pip install rtree")

    models_to_run = ["siren", "mfn", "wire", "incode", "fr"]
    
    with open('results_3d.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Time(s)', 'Note'])

    for mname in models_to_run:
        try:
            # Batch size 16384 uses both T4s efficiently
            time_taken = train_occupancy(mname, MESH_PATH, epochs=20, batch_size=16384)
            
            with open('results_3d.csv', 'a', newline='') as f:
                csv.writer(f).writerow([mname, f"{time_taken:.2f}", "Saved to outputs_3d/"])
                
        except Exception as e:
            print(f"Failed on {mname}: {e}")
            import traceback
            traceback.print_exc()
