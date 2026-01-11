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
    """Calculates Intersection over Union (IoU) as per Section 3.3 of the paper."""
    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()

def extract_mesh(model, resolution=128, threshold=0.5, device='cuda'):
    """Implicit to Explicit conversion using Marching Cubes."""
    model.eval()
    grid_points = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
    points_t = torch.from_numpy(points).float().to(device)
    
    outputs = []
    chunk_size = 50000 
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

def train_occupancy(model_name, dataset_path, epochs=20, batch_size=4096, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Parameter Extraction for Insights
    cfg = BEST_CONFIGS_3D[model_name].copy()
    params_str = str(cfg) 
    hidden_layers = cfg.pop("hidden_layers", 4)
    
    model = create_model(model_name, in_features=3, out_features=1, hidden_layers=hidden_layers, **cfg).to(device)
    dataset = OccupancyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() 

    # Prepare for Per-Epoch Logs
    log_dir = "logs_3d"
    os.makedirs(log_dir, exist_ok=True)
    epoch_log_file = f"{log_dir}/{model_name}_training.csv"
    
    with open(epoch_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'iou'])

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
        # [cite_start]Added model name to the epoch printout for clarity [cite: 13]
        print(f"[{model_name.upper()}] Epoch {epoch+1:02d} | Loss: {avg_loss:.5f} | IoU: {avg_iou:.4f}")
        
        with open(epoch_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss, avg_iou])

    duration = time.time() - start_time
    rec_mesh = extract_mesh(model, resolution=128 if torch.cuda.is_available() else 64, device=device)
    
    if rec_mesh:
        os.makedirs("outputs_3d", exist_ok=True)
        mesh_name = os.path.basename(dataset_path).replace("_dataset.npz", "")
        out_path = f"outputs_3d/{model_name}_{mesh_name}.obj"
        rec_mesh.export(out_path)

    return duration, avg_iou, params_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="nefertiti.obj")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    base_name = os.path.splitext(args.mesh)[0]
    dataset_file = f"{base_name}_dataset.npz"

    models_to_run = ["incode", "fr", "finer", "wire", "fourier", "gauss", "mfn", "siren"]
    csv_file = 'results_3d_comparison.csv'
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Model', 'Mesh', 'Time(s)', 'Final_IoU', 'Config_Params'])

    print(f"\n--- Starting 3D Occupancy Benchmark on: {args.mesh} ---")

    for mname in models_to_run:
        try:
            # [cite_start]Print current model name before training starts [cite: 13]
            print(f"\n>> NOW TRAINING: {mname.upper()}") 
            
            bs = 16384 if torch.cuda.is_available() else 2048
            time_taken, final_iou, config_str = train_occupancy(mname, dataset_file, epochs=args.epochs, batch_size=bs)
            
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([mname, base_name, f"{time_taken:.2f}", f"{final_iou:.4f}", config_str])
                
        except Exception as e:
            print(f"[FAIL] {mname} crashed: {e}")
            traceback.print_exc()
