import os
import time
import torch
import csv
import argparse
import traceback
import gc
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import lpips
from src.models import create_model
from src.trainer import train_inr_for_scale
from src.utils import save_tensor_image, save_rgb_image, save_comparison_grid
from src.config import BEST_CONFIGS

def load_and_process_image(image_path, scale):
    # 1. Load HR
    img = Image.open(image_path).convert("RGB")
    
    w, h = img.size
    w = w - (w % 2)
    h = h - (h % 2)
    img = img.crop((0, 0, w, h))
    
    hr_tensor = transforms.ToTensor()(img) # [C, H, W]
    
    # 2. Create LR (Downsample) using bicubic to match standard datasets
    lr_tensor = F.interpolate(
        hr_tensor.unsqueeze(0), 
        scale_factor=1/scale, 
        mode='bicubic', 
        align_corners=False, 
        antialias=True
    ).squeeze(0)
    
    return hr_tensor.clamp(0, 1), lr_tensor.clamp(0, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run INR Super-Resolution")
    parser.add_argument("--image", type=str, required=True, help="Path to high-res image")
    parser.add_argument("--scales", type=int, nargs='+', default=[4, 8], help="Scales to test")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DYNAMIC BATCH SIZE (Matches Experiment Logic) ---
    BATCH_SIZE = 2000
    if device.type == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        if gpu_mem > 16000: BATCH_SIZE = 10000
        elif gpu_mem > 8000: BATCH_SIZE = 5000
    print(f"Batch Size set to: {BATCH_SIZE}")

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        exit()

    img_name = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs("outputs_2d", exist_ok=True)
    
    try:
        lpips_fn = lpips.LPIPS(net='vgg').to('cpu').eval()
    except:
        print("Warning: LPIPS failed to load.")
        lpips_fn = None

    # Models to run
    model_order = ["gauss", "fourier"]
    
    # UPDATED: LR to match experiment exactly
    DEFAULT_LR = 1e-4 
    DEFAULT_LOSS = "mse"

    summary_history = []
    
    print(f"\n{'='*70}\nPROCESSING: {args.image}\n{'='*70}")

    for scale in args.scales:
        print(f"\n{'='*70}\nSCALE {scale}×\n{'='*70}")
        
        hr_tensor, lr_tensor = load_and_process_image(args.image, scale)
        
        print(f"HR: {hr_tensor.shape[1]}x{hr_tensor.shape[2]} | LR: {lr_tensor.shape[1]}x{lr_tensor.shape[2]}")
        
        results = {} 

        for mname in model_order:
            cfg = BEST_CONFIGS.get(mname, {}).copy()
            hidden_layers = cfg.pop("hidden_layers", 4)
            cfg_str = ", ".join([f"{k}={v}" for k, v in cfg.items()])
            
            print(f"\n{'-'*70}\n{mname.upper()} @ {scale}×\n{'-'*70}", flush=True)

            model = None
            pred_rgb = None
            
            try:
                model = create_model(mname, hidden_layers=hidden_layers, **cfg)
                start_t = time.time()
                
                pred_rgb, psnr_val, ssim_val, lpips_val = train_inr_for_scale(
                    model=model,
                    hr_tensor=hr_tensor,
                    lr_tensor=lr_tensor,
                    scale=scale,
                    model_name=mname,
                    config_str=cfg_str,
                    epochs=args.epochs,
                    lr=DEFAULT_LR,        # Now 1e-4
                    batch_size=BATCH_SIZE, # Now Dynamic (likely 5000-10000)
                    device=device,
                    lpips_fn=lpips_fn,
                    loss_type=DEFAULT_LOSS,
                    checkpoint_dir=f'checkpoints_{mname}_{scale}x',
                    csv_path='results_2d.csv'
                )
                
                train_duration = time.time() - start_t

                pred_out = os.path.join("outputs_2d", f"{img_name}_{mname}_{scale}x.png")
                save_rgb_image(pred_rgb, pred_out)

                results[mname] = {'pred': pred_rgb, 'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}
                
                summary_history.append({
                    'Image': img_name, 'Model': mname, 'Scale': scale, 
                    'PSNR': psnr_val, 'SSIM': ssim_val, 'LPIPS': lpips_val, 'Time(s)': train_duration
                })

            except Exception as e:
                print(f"[ERROR] {mname} @ {scale}× failed: {e}", flush=True)
                traceback.print_exc()
            finally:
                del pred_rgb, model
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        try:
            grid_out = os.path.join("outputs_2d", f"{img_name}_ALL_MODELS_{scale}x_grid.png")
            save_comparison_grid(hr_tensor, results, grid_out, cols=3)
        except Exception as e:
            print(f"[WARN] Grid failed: {e}")

    if summary_history:
        with open('final_summary_2d.csv', 'w', newline='') as f:
            fieldnames = ['Image', 'Model', 'Scale', 'PSNR', 'SSIM', 'LPIPS', 'Time(s)']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_history:
                row.update({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
                writer.writerow(row)
        print("\nDone. Summary saved to final_summary_2d.csv")
