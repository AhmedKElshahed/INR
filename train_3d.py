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
    
    # Query model in chunks to save VRAM
    outputs = []
    chunk_size = 50000
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

def train_occupancy(model_name, mesh_path, epochs=50, batch_size=4096, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on 3D Occupancy")
    print(f"{'='*60}")

    # 1. Config & Model
    # Note: in_features=3 (x,y,z), out_features=1 (occupancy probability)
    cfg = BEST_CONFIGS[model_name].copy()
    hidden_layers = cfg.pop("hidden_layers", 4)
    model = create_model(model_name, in_features=3, out_features=1, hidden_layers=hidden_layers, **cfg)
    model = model.to(device)

    # 2. Data
    # We sample 2 million points for training to get good detail
    dataset = OccupancyDataset(mesh_path, num_samples=200000, on_surface_ratio=0.5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # MSE works well for occupancy probabilities [0,1]

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
            
            # Ensure outputs are in [0,1] range for IoU calculation logic (optional for loss depending on model output)
            # Some INRs output logits, but our MSE approach assumes raw output approximates 0 or 1.
            # If model output is unbounded, you might want torch.sigmoid(preds) here, 
            # but standard SIREN/MFN usually train directly on regression targets.
            
            loss = criterion(preds, t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                # Calculate IoU on training batch
                total_iou += calculate_iou(preds, t)
            steps += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/steps:.5f} | Train IoU: {total_iou/steps:.4f}")

    duration = time.time() - start_time

    # 5. Extract and Save Mesh
    os.makedirs("outputs_3d", exist_ok=True)
    print("Extracting mesh via Marching Cubes (resolution 256)...")
    
    # Higher resolution = better detail but slower
    rec_mesh = extract_mesh(model, resolution=128, device=device)
    
    if rec_mesh:
        out_path = f"outputs_3d/{model_name}_dragon.obj"
        rec_mesh.export(out_path)
        print(f"Saved mesh to {out_path}")
    
    return duration

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # PATH TO YOUR INPUT MESH (change this)
    MESH_PATH = "xyzrgb_dragon.obj" 
    
    if not os.path.exists(MESH_PATH):
        print(f"Error: Could not find {MESH_PATH}. Please download a mesh (e.g. Stanford Dragon) first.")
        exit()

    models_to_run = ["siren", "mfn", "wire", "incode", "fr"] # Add others as needed
    
    # CSV logging
    with open('results_3d.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Time(s)', 'Note'])

    for mname in models_to_run:
        try:
            time_taken = train_occupancy(mname, MESH_PATH, epochs=20, batch_size=2048)
            
            with open('results_3d.csv', 'a', newline='') as f:
                csv.writer(f).writerow([mname, f"{time_taken:.2f}", "Saved to outputs_3d/"])
                
        except Exception as e:
            print(f"Failed on {mname}: {e}")
            import traceback
            traceback.print_exc()
