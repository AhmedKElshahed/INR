import torch
import torch.nn as nn
import os
import csv
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from .utils import get_mgrid, compute_metrics

def train_inr_for_scale(
    model, hr_tensor, lr_tensor, scale,
    model_name, config_str,
    epochs=100, lr=1e-4, batch_size=2000,
    device='cuda', lpips_fn=None, csv_path='results.csv',
    checkpoint_dir='checkpoints', grad_clip=1.0,
    loss_type="mse"
):
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} for {scale}× Super-Resolution")
    print(f"Config: {config_str}")
    print(f"Loss: {loss_type.upper()}")
    print(f"{'='*70}\n")
    
    c, h, w = lr_tensor.shape
    coords = get_mgrid(h, w)
    rgb = lr_tensor.permute(1, 2, 0).reshape(-1, 3)
    
    model = model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"-> Activating DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    dataset = TensorDataset(coords, rgb)
    pin = (device.type == 'cuda') and torch.cuda.is_available()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=0.01)
    else:
        raise ValueError("loss_type must be 'mse' or 'smoothl1'")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            for p in model.parameters():
                if p.grad is not None and torch.is_complex(p.grad):
                    p.grad = p.grad.resolve_conj().clone()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=False)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(1, len(dataloader))
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'config': config_str
            }, ckpt_path)
    
    print("\nEvaluating on HR resolution...")
    if device.type == 'cuda': torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        hr_h, hr_w = hr_tensor.shape[1], hr_tensor.shape[2]
        hr_coords = get_mgrid(hr_h, hr_w).to(device)
        pred_rgb = []
        chunk_size = 20000
        total = hr_coords.shape[0]
        for i in range(0, total, chunk_size):
            coords_chunk = hr_coords[i:i+chunk_size]
            pred_chunk = model(coords_chunk)
            pred_rgb.append(pred_chunk.cpu())
            if (i // chunk_size) % 10 == 0:
                print(f"  [{i+len(coords_chunk)}/{total}] pixels done...")

        pred_rgb = torch.cat(pred_rgb, dim=0)
        pred_rgb = pred_rgb.clamp(0, 1).numpy().reshape(hr_h, hr_w, 3)
    
    psnr_val, ssim_val, lpips_val = compute_metrics(hr_tensor, pred_rgb, lpips_fn, device)
    score = psnr_val + 10 * ssim_val - 5 * lpips_val
    
    print(f"FINAL: PSNR {psnr_val:.2f} | SSIM {ssim_val:.4f} | LPIPS {lpips_val:.4f} | Score {score:.2f}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['timestamp', 'model', 'scale', 'config', 'psnr', 'ssim', 'lpips', 'score', 'epochs', 'lr', 'loss'])
        writer.writerow([timestamp, model_name, scale, config_str, f"{psnr_val:.2f}", f"{ssim_val:.4f}", f"{lpips_val:.4f}", f"{score:.2f}", epochs, lr, loss_type])
    
    final_path = os.path.join(checkpoint_dir, f"{model_name}_{scale}x_final.pth")
    torch.save({'model_state_dict': model.state_dict(), 'config': config_str}, final_path)
    
    return pred_rgb, psnr_val, ssim_val, lpips_val
