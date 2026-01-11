import os
import time
import torch
import torch.nn as nn
import numpy as np
import csv
import argparse
import traceback
from torch.utils.data import DataLoader
import skimage.measure
import trimesh

# Fix OpenMP conflict on Windows/Conda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models import create_model
from src.data_3d import OccupancyDataset
from src.config import BEST_CONFIGS_3D

# ============================================================================
# 3D UTILS
# ============================================================================

def calculate_iou(pred_logits, target, threshold=0.5):
    """Calculates Intersection over Union (IoU) for occupancy."""
    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()

def evaluate_full_grid_iou(model, mesh_path, resolution=128, threshold=0.5, device='cuda'):
    """
    Calculates the 'True IoU' by comparing a structured grid of model predictions
    against the ground-truth mesh occupancy. Matches Table 4 methodology[cite: 1378].
    """
    model.eval()
    
    # 1. Load and normalize the ground truth mesh (same as generate_data.py)
    mesh = trimesh.load(mesh_path)
    vertices = mesh.vertices
    center = (vertices.max(0) + vertices.min(0)) / 2
    scale = 1.8 / (vertices.max(0) - vertices.min(0)).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)

    # 2. Create a structured 3D grid
    grid_points = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
    
    # 3. Get Ground Truth Occupancy
    gt_occ = mesh.contains(points).astype(np.float32)
    gt_occ_t = torch.from_numpy(gt_occ).to(device).unsqueeze(-1)

    # 4. Get Model Predictions in chunks
    points_t = torch.from_numpy(points).float().to(device)
    preds = []
    chunk_size = 100000
    with torch.no_grad():
        for i in range(0, len(points_t), chunk_size):
            batch = points_t[i:i+chunk_size]
            out = model(batch)
            preds.append(out)
    
    pred_logits = torch.cat(preds, dim=0)
    
    # 5. Calculate Final IoU
    return calculate_iou(pred_logits, gt_occ_t, threshold)

def extract_mesh(model, resolution=128, threshold=0.5, device='cuda'):
    """Implicit to Explicit conversion using Marching Cubes."""
    model.eval()
    grid_points = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
    points_t = torch.from_numpy(points).float().to(device)
    
    outputs = []
    chunk_size = 100000 
    with torch.no_grad():
        for i in range(0, len(points_t), chunk_size):
            batch = points_t[i:i+chunk_size]
            out = model(batch)
            outputs.append(out.cpu())
            
    outputs = torch.sigmoid(torch.cat(outputs, dim=0)).numpy().reshape(resolution, resolution, resolution)
    
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(outputs, level=threshold)
        verts = verts / (resolution - 1) * 2 - 1
        return trimesh.Trimesh(vertices=verts, faces=faces)
    except ValueError:
        return None

# ============================================================================
# TRAINING ENGINE
# ============================================================================

def train_occupancy(model_name, mesh_path, dataset_path, epochs=20, batch_size=4096, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = BEST_CONFIGS_3D[model_name].copy()
    params_str = str(cfg) 
    hidden_layers = cfg.pop("hidden_layers", 4)
    
    model = create_model(model_name, in_features=3, out_features=1, hidden_layers=hidden_layers, **cfg).to(device)
    dataset = OccupancyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() 

    log_dir = "logs_3d"
    os.makedirs(log_dir, exist_ok=True)
    epoch_log_file = f"{log_dir}/{model_name}_training.csv"
    
    with open(epoch_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'train_iou'])

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss, total_iou, steps = 0, 0, 0
        for p, t in dataloader:
            p, t = p.to(device), t.to(device)
            optimizer.zero_grad()
            preds = model(p)
            loss = criterion(preds, t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_iou += calculate_iou(preds, t)
            steps += 1
        
        avg_loss = total_loss/steps
        avg_iou = total_iou/steps
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.5f} | Train IoU: {avg_iou:.4f}")
        
        with open(epoch_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss, avg_iou])

    duration = time.time() - start_time
    
    # --- PAPER ACCURATE EVALUATION ---
    print(f"-> Running final grid-based evaluation for {model_name}...")
    final_grid_iou = evaluate_full_grid_iou(model, mesh_path, resolution=128, device=device)
    print(f"-> Final Grid IoU: {final_grid_iou:.4f}")

    # Mesh Extraction
    rec_mesh = extract_mesh(model, resolution=128 if torch.cuda.is_available() else 64, device=device)
    if rec_mesh:
        os.makedirs("outputs_3d", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        out_path = f"outputs_3d/{model_name}_{base_name}.obj"
        rec_mesh.export(out_path)

    return duration, final_grid_iou, params_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="nefertiti.obj")
    parser.add_argument("--epochs", type=int, default=30) # Increased default for better convergence
    args = parser.parse_args()
    
    base_name = os.path.splitext(args.mesh)[0]
    dataset_file = f"{base_name}_dataset.npz"

    models_to_run = ["finer", "incode", "siren", "wire", "fourier", "gauss", "fr", "mfn"]
    csv_file = 'results_3d_comparison.csv'
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Model', 'Mesh', 'Time(s)', 'Final_Grid_IoU', 'Config_Params'])

    for mname in models_to_run:
        try:
            bs = 16384 if torch.cuda.is_available() else 2048
            time_taken, grid_iou, config_str = train_occupancy(mname, args.mesh, dataset_file, epochs=args.epochs, batch_size=bs)
            
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([mname, base_name, f"{time_taken:.2f}", f"{grid_iou:.4f}", config_str])
                
        except Exception as e:
            print(f"[FAIL] {mname} crashed: {e}")
            traceback.print_exc()
