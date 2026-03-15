import os
import time
import torch
import torch.nn as nn
import numpy as np
import csv
import argparse
import traceback
from torch.utils.data import DataLoader, random_split
import skimage.measure
import trimesh

# Fix OpenMP conflict on Windows/Conda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models import create_model
from src.data_3d import OccupancyDataset
from src.config import BEST_CONFIGS_3D

# Some activations have smaller or less-stable gradients; override LR for stable 3D convergence.
MODEL_LR_OVERRIDES = {
    'gauss': 3e-4,   # 1e-3 caused erratic training on 3D near-surface-heavy data
    'mfn':   3e-4,   # plateaus at train IoU 0.91 at default 1e-4; needs higher LR
}

# ============================================================================
# 3D METRIC UTILS
# ============================================================================

def calculate_iou(pred_logits, target, threshold=0.5):
    """Intersection over Union for binary occupancy (logit inputs)."""
    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def extract_mesh(model, resolution=128, threshold=0.5, device='cuda'):
    """Implicit -> Explicit via Marching Cubes."""
    model.eval()
    grid_points = np.linspace(-1, 1, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
    points_t = torch.from_numpy(points).float().to(device)

    outputs = []
    chunk_size = 50000
    with torch.no_grad():
        for i in range(0, len(points_t), chunk_size):
            batch = points_t[i:i + chunk_size]
            out = model(batch)
            outputs.append(out.cpu())

    outputs = torch.sigmoid(torch.cat(outputs, dim=0)).numpy().reshape(resolution, resolution, resolution)

    # Smooth the volume to reduce marching-cubes staircase artifacts
    from scipy.ndimage import gaussian_filter
    outputs = gaussian_filter(outputs, sigma=1.0)

    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(outputs, level=threshold)
        verts = verts / (resolution - 1) * 2 - 1
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # Keep only the largest connected component — removes floating artifacts
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda m: len(m.faces))
        # Laplacian smoothing to remove residual voxel-stepping on the surface
        trimesh.smoothing.filter_laplacian(mesh, iterations=3)
        return mesh
    except ValueError:
        return None


def load_gt_mesh_normalized(mesh_path, norm_center, norm_max_dist):
    """
    Load the original .obj mesh and apply the same two-step normalization
    used during data generation (generate_data.py + data_3d.py), so its
    coordinate space matches the marching-cubes predicted mesh.
    """
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
    except Exception as e:
        print(f"  [WARN] Could not load GT mesh: {e}")
        return None

    verts = np.asarray(mesh.vertices, dtype=np.float64)

    # Step 1: generate_data.py normalization (-> ≈ [-0.9, 0.9])
    c1 = (verts.max(0) + verts.min(0)) / 2
    s1 = 1.8 / (verts.max(0) - verts.min(0)).max()
    verts = (verts - c1) * s1

    # Step 2: data_3d.py normalization using the stored dataset params (-> [-1, 1])
    verts = verts - norm_center.astype(np.float64)
    if norm_max_dist > 0:
        verts = verts / norm_max_dist

    return trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces))


def compute_chamfer_and_nc(gt_mesh, pred_mesh, n_samples=10000):
    """
    Chamfer L1 Distance and Normal Consistency between two meshes.
    Both metrics follow: Occupancy Networks (Mescheder et al., CVPR 2019).
      - Chamfer L1 = mean of (accuracy: pred->GT) + (completeness: GT->pred)
      - Normal Consistency = mean |cos(angle)| between matched normals
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  [WARN] scipy not installed — skipping Chamfer/NC. Run: pip install scipy")
        return float('nan'), float('nan')

    try:
        gt_pts,   gt_fidx   = trimesh.sample.sample_surface(gt_mesh,   n_samples)
        pred_pts, pred_fidx = trimesh.sample.sample_surface(pred_mesh, n_samples)

        gt_normals   = gt_mesh.face_normals[gt_fidx]
        pred_normals = pred_mesh.face_normals[pred_fidx]

        tree_gt   = cKDTree(gt_pts)
        tree_pred = cKDTree(pred_pts)

        d_pred2gt, idx_p2g = tree_gt.query(pred_pts)   # accuracy
        d_gt2pred, _       = tree_pred.query(gt_pts)   # completeness

        chamfer = (d_pred2gt.mean() + d_gt2pred.mean()) / 2.0

        matched_gt_normals = gt_normals[idx_p2g]
        nc = float(np.abs((pred_normals * matched_gt_normals).sum(axis=-1)).mean())

        return float(chamfer), nc

    except Exception as e:
        print(f"  [WARN] Chamfer/NC computation failed: {e}")
        return float('nan'), float('nan')


def compute_eval_iou(model, val_loader, device):
    """IoU on the held-out validation split (not seen during training)."""
    model.eval()
    total_iou, steps = 0.0, 0
    with torch.no_grad():
        for p, t in val_loader:
            p, t = p.to(device), t.to(device)
            preds = model(p)
            total_iou += calculate_iou(preds, t)
            steps += 1
    return total_iou / max(steps, 1)


# ============================================================================
# TRAINING ENGINE
# ============================================================================

def train_occupancy(model_name, dataset_path, mesh_path=None,
                    epochs=500, batch_size=4096, lr=1e-4, args_res=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = BEST_CONFIGS_3D[model_name].copy()
    params_str = str(cfg)
    hidden_layers = cfg.pop("hidden_layers", 4)

    model = create_model(model_name, in_features=3, out_features=1,
                         hidden_layers=hidden_layers, **cfg).to(device)

    # ---- Dataset: 90% train / 10% val split ----
    dataset = OccupancyDataset(dataset_path)
    val_size   = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Optimizer + cosine annealing LR schedule ----
    effective_lr = MODEL_LR_OVERRIDES.get(model_name, lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=effective_lr * 0.01
    )
    criterion = nn.BCEWithLogitsLoss()

    # ---- Per-epoch log ----
    log_dir = "logs_3d"
    os.makedirs(log_dir, exist_ok=True)
    epoch_log = f"{log_dir}/{model_name}_training.csv"
    with open(epoch_log, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_iou', 'lr'])

    lr_note = f"  (override, default={lr:.0e})" if model_name in MODEL_LR_OVERRIDES else ""
    print(f"\n{'='*60}")
    print(f"  Model : {model_name.upper()}")
    print(f"  Device: {device}  |  Epochs: {epochs}  |  BS: {batch_size}")
    print(f"  LR    : {effective_lr:.0e}{lr_note}")
    print(f"  Train : {train_size} pts   |  Val: {val_size} pts")
    print(f"{'='*60}")

    start_time = time.time()
    avg_loss = avg_iou = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, total_iou, steps = 0.0, 0.0, 0

        for p, t in train_loader:
            p, t = p.to(device), t.to(device)
            optimizer.zero_grad()
            preds = model(p)
            loss = criterion(preds, t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_iou  += calculate_iou(preds, t)
            steps += 1

        avg_loss = total_loss / steps
        avg_iou  = total_iou  / steps
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Print every 10 epochs or at the very end
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"[{model_name.upper():8s}] Epoch {epoch+1:03d}/{epochs} "
                  f"| Loss: {avg_loss:.5f} | Train IoU: {avg_iou:.4f} | LR: {current_lr:.2e}")

        with open(epoch_log, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_loss:.6f}", f"{avg_iou:.5f}", f"{current_lr:.2e}"])

    duration = time.time() - start_time

    # ---- Post-training: eval IoU on held-out validation split ----
    print(f"\n[{model_name.upper()}] Computing eval IoU on validation split...")
    eval_iou = compute_eval_iou(model, val_loader, device)
    print(f"[{model_name.upper()}] Eval IoU: {eval_iou:.4f}")

    # ---- Mesh extraction (Marching Cubes) ----
    res = args_res if args_res is not None else (256 if torch.cuda.is_available() else 64)
    print(f"[{model_name.upper()}] Extracting mesh (resolution={res})...")
    os.makedirs("outputs_3d", exist_ok=True)
    mesh_stem = os.path.basename(dataset_path).replace("_dataset.npz", "")
    out_path = f"outputs_3d/{model_name}_{mesh_stem}.obj"

    pred_mesh = extract_mesh(model, resolution=res, device=device)
    if pred_mesh:
        pred_mesh.export(out_path)
        print(f"[{model_name.upper()}] Saved mesh -> {out_path}")
    else:
        print(f"[{model_name.upper()}] Warning: no surface extracted.")

    # ---- Chamfer Distance + Normal Consistency (vs GT mesh) ----
    chamfer, nc = float('nan'), float('nan')
    if pred_mesh and mesh_path and os.path.exists(mesh_path):
        print(f"[{model_name.upper()}] Computing Chamfer L1 & Normal Consistency...")
        gt_mesh = load_gt_mesh_normalized(mesh_path, dataset.norm_center, dataset.norm_max_dist)
        if gt_mesh is not None:
            chamfer, nc = compute_chamfer_and_nc(gt_mesh, pred_mesh, n_samples=10000)
            print(f"[{model_name.upper()}] Chamfer L1: {chamfer:.6f}  |  Normal Consistency: {nc:.4f}")

    return duration, avg_iou, eval_iou, chamfer, nc, params_str


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Occupancy INR benchmark — trains all models on a mesh and reports IoU + Chamfer."
    )
    parser.add_argument("--mesh",   type=str,   default="nefertiti.obj",
                        help="Input .obj mesh file")
    parser.add_argument("--epochs", type=int,   default=500,
                        help="Training epochs per model (500+ recommended with near-surface-heavy dataset)")
    parser.add_argument("--lr",     type=float, default=1e-4,
                        help="Initial learning rate (cosine-annealed to lr*0.01)")
    parser.add_argument("--res",    type=int,   default=None,
                        help="Marching-cubes resolution (default: 256 on GPU, 64 on CPU)")
    args = parser.parse_args()

    base_name    = os.path.splitext(args.mesh)[0]
    dataset_file = f"{base_name}_dataset.npz"

    if not os.path.exists(dataset_file):
        print(f"[ERROR] '{dataset_file}' not found.")
        print(f"  Run: python generate_data.py --mesh {args.mesh}")
        exit(1)

    models_to_run = ["siren", "wire", "finer", "gauss", "mfn", "fourier", "incode", "fr"]

    csv_file = 'results_3d_comparison.csv'
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        if write_header:
            csv.writer(f).writerow([
                'Model', 'Mesh', 'Epochs', 'Time(s)',
                'Final_Train_IoU', 'Eval_IoU',
                'Chamfer_L1', 'Normal_Consistency',
                'Config_Params'
            ])

    print(f"\n=== 3D Occupancy Benchmark ===")
    print(f"  Mesh   : {args.mesh}")
    print(f"  Epochs : {args.epochs}")
    print(f"  LR     : {args.lr}")
    print(f"  Models : {models_to_run}")
    print(f"  Output : {csv_file}")

    for mname in models_to_run:
        try:
            print(f"\n{'='*60}")
            print(f">> STARTING: {mname.upper()}")
            bs = 16384 if torch.cuda.is_available() else 2048

            duration, train_iou, eval_iou, chamfer, nc, cfg_str = train_occupancy(
                mname, dataset_file,
                mesh_path=args.mesh,
                epochs=args.epochs,
                batch_size=bs,
                lr=args.lr,
                args_res=args.res,
            )

            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    mname, base_name, args.epochs,
                    f"{duration:.1f}",
                    f"{train_iou:.4f}",
                    f"{eval_iou:.4f}",
                    f"{chamfer:.6f}" if not np.isnan(chamfer) else 'N/A',
                    f"{nc:.4f}"      if not np.isnan(nc)      else 'N/A',
                    cfg_str,
                ])

        except Exception as e:
            print(f"[FAIL] {mname} crashed: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done. Results -> {csv_file} | Meshes -> outputs_3d/")
