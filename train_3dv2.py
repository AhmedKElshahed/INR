import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
import time
import trimesh
from skimage import measure
from scipy.spatial import cKDTree

# --- METRIC UTILITIES (Matching Paper Methodology) ---

def calculate_iou(pred_logits, gt_occupancy):
    """Calculates Intersection over Union for 3D volumes."""
    pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred_binary * gt_occupancy).sum()
    union = (pred_binary + gt_occupancy).clamp(0, 1).sum()
    return (intersection / (union + 1e-6)).item()

def calculate_chamfer_dist(model, gt_mesh, resolution=128, device='cuda'):
    """Calculates Chamfer Distance by extracting the mesh mid-train."""
    model.eval()
    # 1. Create Grid
    grid_coords = torch.meshgrid(torch.linspace(-1, 1, resolution),
                                 torch.linspace(-1, 1, resolution),
                                 torch.linspace(-1, 1, resolution), indexing='ij')
    coords = torch.stack(grid_coords, dim=-1).reshape(-1, 3).to(device)
    
    # 2. Query Model
    with torch.no_grad():
        logits = model(coords)
        volume = torch.sigmoid(logits).reshape(resolution, resolution, resolution).cpu().numpy()
    
    # 3. Marching Cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
        # Normalize verts back to [-1, 1]
        verts = (verts / resolution) * 2 - 1
        recon_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # 4. Sample and Compute Distance
        p_recon = recon_mesh.sample(10000)
        p_gt = gt_mesh.sample(10000)
        
        one_way_dist = cKDTree(p_gt).query(p_recon)[0]
        other_way_dist = cKDTree(p_recon).query(p_gt)[0]
        return np.mean(one_way_dist**2) + np.mean(other_way_dist**2), len(verts)
    except:
        return 1.0, 0 # Return high error if no surface found

# --- MAIN TRAINING SCRIPT ---

def train_occupancy(args, model, train_loader, gt_mesh):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # SCIENTIFIC ADJUSTMENT: Paper uses BCE for occupancy classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Insight Export Setup
    log_file = f"insights_{args.model_type}.csv"
    
    print(f"Starting optimized training for {args.model_type}...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_iou = 0
        
        for batch_coords, batch_occupancy in train_loader:
            batch_coords, batch_occupancy = batch_coords.to(device), batch_occupancy.to(device)
            
            optimizer.zero_grad()
            output = model(batch_coords)
            
            # Binary Cross Entropy Loss
            loss = criterion(output, batch_occupancy)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_iou += calculate_iou(output, batch_occupancy)
            
        # Periodic Geometric Insights (Every 10 epochs to save time)
        if epoch % 10 == 0:
            chamfer, v_count = calculate_chamfer_dist(model, gt_mesh, device=device)
            avg_iou = epoch_iou / len(train_loader)
            avg_loss = epoch_loss / len(train_loader)
            
            # Export to CSV
            file_exists = os.path.isfile(log_file)
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'bce_loss', 'iou', 'chamfer', 'verts'])
                if not file_exists: writer.writeheader()
                writer.writerow({
                    'epoch': epoch,
                    'bce_loss': f"{avg_loss:.6f}",
                    'iou': f"{avg_iou:.4f}",
                    'chamfer': f"{chamfer:.8f}",
                    'verts': v_count
                })
            
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Chamfer: {chamfer:.6f}")

    # Final Export
    model_name = f"{args.model_type}_final.obj"
    export_mesh(model, model_name) # Call existing export function
