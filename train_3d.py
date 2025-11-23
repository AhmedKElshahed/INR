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
from src.config import BEST_CONFIGS

# ============================================================================
# 3D UTILS
# ============================================================================

def calculate_iou(pred_logits, target, threshold=0.5):
    """
    Calculates Intersection over Union (IoU) for binary occupancy.
    Expects raw logits from the model, applies Sigmoid internally.
    """
    # Convert Logits (-inf, inf) -> Probabilities (0, 1)
    probs = torch.sigmoid(pred_logits)
    
    pred_bin = (probs > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    
    return (intersection / (union + 1e-6)).item()

def extract_mesh(model, resolution=128, threshold=0.5, device='cuda'):
    """
    Runs Marching Cubes to extract the mesh surface from the INR.
    """
    model.eval()
    
    # Create a 3D grid of query points
    grid_points = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
    points_t = torch.from_numpy(points).float().to(device)
    
    outputs = []
    chunk_size = 50000 # Process in chunks to save VRAM
    
    with torch.no_grad():
        for i in range(0, len(points_t), chunk_size):
            batch = points_t[i:i+chunk_size]
            out = model(batch)
            outputs.append(out.cpu())
            
    # Concatenate chunks
    outputs = torch.cat(outputs, dim=0)
    
    # Apply Sigmoid (Logits -> Probabilities)
    outputs = torch.sigmoid(outputs)
    
    # Reshape to 3D volume
    outputs = outputs.numpy().reshape(resolution, resolution, resolution)
    
    # Marching Cubes algorithm to find the isosurface at threshold (0.5)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(outputs, level=threshold)
        # Normalize vertices back to [-1, 1] range
        verts = verts / (resolution - 1) * 2 - 1
        return trimesh.Trimesh(vertices=verts, faces=faces)
    except ValueError:
        print("Warning: No surface found at this threshold.")
        return None

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_occupancy(model_name, dataset_path, epochs=20, batch_size=4096, lr=1e-4):
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        gpu_count = 0
        device_name = "CPU"

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"Hardware: {device_name} (Count: {gpu_count})")
    print(f"{'='*60}")

    # 2. Model Initialization
    cfg = BEST_CONFIGS[model_name].copy()
    hidden_layers = cfg.pop("hidden_layers", 4)
    
    # in_features=3 (x,y,z), out_features=1 (occupancy probability)
    model = create_model(model_name, in_features=3, out_features=1, hidden_layers=hidden_layers, **cfg)
    
    model = model.to(device)
    if gpu_count > 1:
        print(f"-> Activating DataParallel on {gpu_count} GPUs")
        model = nn.DataParallel(model)

    # 3. Data Loading
    # Note: Loads the pre-processed .npz file
    dataset = OccupancyDataset(dataset_path)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        # Set workers to 0 on Windows/CPU to prevent multiprocessing errors
        num_workers=0 if (os.name == 'nt' or gpu_count == 0) else 2, 
        pin_memory=(gpu_count > 0)
    )

    # 4. Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # BCEWithLogitsLoss is more stable than BCE + Sigmoid
    criterion = nn.BCEWithLogitsLoss() 

    # 5. Training Loop
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_iou = 0
        steps = 0
        
        for p, t in dataloader:
            p, t = p.to(device), t.to(device)
            
            optimizer.zero_grad()
            preds = model(p) # Model outputs raw logits
            
            loss = criterion(preds, t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                total_iou += calculate_iou(preds, t)
            steps += 1
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/steps:.5f} | Train IoU: {total_iou/steps:.4f}")

    duration = time.time() - start_time

    # 6. Reconstruction (Mesh Extraction)
    os.makedirs("outputs_3d", exist_ok=True)
    
    # Resolution: Higher is better but slower. 
    # 64 is fast (CPU friendly), 128/256 is high detail (GPU friendly)
    res = 128 if gpu_count > 0 else 64
    print(f"-> Extracting mesh (Resolution: {res})...")
    
    rec_mesh = extract_mesh(model, resolution=res, device=device)
    
    if rec_mesh:
        # Output naming: siren_dragon.obj
        mesh_name = os.path.basename(dataset_path).replace("_dataset.npz", "")
        out_path = f"outputs_3d/{model_name}_{mesh_name}.obj"
        rec_mesh.export(out_path)
        print(f"-> Saved mesh to {out_path}")
    
    return duration

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="dragon.obj", help="Input .obj file name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()
    
    # Derive .npz path from .obj name
    base_name = os.path.splitext(args.mesh)[0]
    dataset_file = f"{base_name}_dataset.npz"
    
    # Check if data exists
    if not os.path.exists(dataset_file):
        print(f"[ERROR] Data file '{dataset_file}' not found.")
        print(f"Please run: python generate_data.py --mesh {args.mesh}")
        exit()

    # Models to compare
    models_to_run = ["incode", "fr", "finer", "wire", "fourier", "gauss", "mfn", "siren"]
    
    # Initialize CSV Logging
    csv_file = 'results_3d.csv'
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model', 'Mesh', 'Time(s)', 'Epochs', 'Output'])

    print(f"Starting benchmark on {dataset_file}...")

    for mname in models_to_run:
        try:
            # Adjust batch size dynamically based on hardware
            bs = 16384 if torch.cuda.is_available() else 2048
            
            time_taken = train_occupancy(mname, dataset_file, epochs=args.epochs, batch_size=bs)
            
            # Log results
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([mname, base_name, f"{time_taken:.2f}", args.epochs, f"outputs_3d/{mname}_{base_name}.obj"])
                
        except Exception as e:
            print(f"[FAIL] {mname} crashed: {e}")
            traceback.print_exc()

    print("\nDone. Check 'outputs_3d' folder for .obj files.")
