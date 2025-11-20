import os
import time
import gc
import traceback
import torch
import csv
import lpips
from PIL import Image
import torchvision.transforms as transforms

# Env fix must happen before imports that use OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models import create_model
from src.trainer import train_inr_for_scale
from src.utils import save_tensor_image, save_rgb_image, save_comparison_grid
from src.config import BEST_CONFIGS

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------ SETTINGS ------------------
    hr_dir = "DIV2K/DIV2K_train_HR"
    lr_base_dir = "DIV2K/DIV2K_train_LR_bicubic"
    img_id = "0788"
    scales = [4, 8, 16]
    
    # Training Parameters
    DEFAULT_EPOCHS = 100
    DEFAULT_LR = 2e-4
    DEFAULT_BS = 2000
    DEFAULT_LOSS = "mse"
    
    model_order = ["siren", "mfn", "fourier", "gauss", "wire", "finer", "incode", "fr"]
    per_model_train = {} # Add overrides here if needed

    # ------------------ DATA LOADING ------------------
    hr_image_path = os.path.join(hr_dir, f"{img_id}.png")
    hr_image = Image.open(hr_image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    hr_tensor = to_tensor(hr_image)

    lr_tensors = {}
    for scale in scales:
        lr_path = os.path.join(lr_base_dir, f"X{scale}", f"{img_id}x{scale}.png")
        lr_tensors[scale] = to_tensor(Image.open(lr_path).convert("RGB"))

    lpips_fn = lpips.LPIPS(net='vgg').to('cpu').eval()

    os.makedirs("outputs", exist_ok=True)
    hr_out = os.path.join("outputs", f"{img_id}_HR.png")
    save_tensor_image(hr_tensor, hr_out)

    # ------------------ MAIN LOOP ------------------
    summary_history = []
    print("\n" + "="*70 + "\nRUN ALL MODELS at ALL SCALES\n" + "="*70)

    for scale in scales:
        print(f"\n{'='*70}\nSCALE {scale}×\n{'='*70}")
        results = {} 

        for mname in model_order:
            cfg = BEST_CONFIGS[mname].copy()
            hidden_layers = cfg.pop("hidden_layers", 4)
            cfg_str = ", ".join([f"{k}={v}" for k, v in cfg.items()] + [f"hidden_layers={hidden_layers}"])
            
            ovr = per_model_train.get(mname, {})
            epochs, lr = ovr.get("epochs", DEFAULT_EPOCHS), ovr.get("lr", DEFAULT_LR)
            batch_size, loss_type = ovr.get("batch_size", DEFAULT_BS), ovr.get("loss_type", DEFAULT_LOSS)

            print(f"\n{'-'*70}\n{mname.upper()} @ {scale}×\n{'-'*70}", flush=True)

            model, pred_rgb = None, None
            try:
                model = create_model(mname, hidden_layers=hidden_layers, **cfg)
                
                start_t = time.time()
                pred_rgb, psnr_val, ssim_val, lpips_val = train_inr_for_scale(
                    model=model, hr_tensor=hr_tensor, lr_tensor=lr_tensors[scale],
                    scale=scale, model_name=mname, config_str=cfg_str,
                    epochs=epochs, lr=lr, batch_size=batch_size,
                    device=device, lpips_fn=lpips_fn, loss_type=loss_type,
                    checkpoint_dir=f'checkpoints_{mname}_{scale}x'
                )
                train_duration = time.time() - start_t

                pred_out = os.path.join("outputs", f"{img_id}_{mname}_{scale}x.png")
                save_rgb_image(pred_rgb, pred_out)

                results[mname] = {'pred': pred_rgb, 'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}
                
                summary_history.append({
                    'Model': mname, 'Scale': scale, 
                    'PSNR': psnr_val, 'SSIM': ssim_val, 'LPIPS': lpips_val, 
                    'Score': psnr_val + 10*ssim_val - 5*lpips_val,
                    'Time(s)': train_duration
                })

            except Exception as e:
                print(f"[ERROR] {mname} @ {scale}× failed: {e}", flush=True)
                traceback.print_exc()
            finally:
                del pred_rgb, model
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        try:
            grid_out = os.path.join("outputs", f"{img_id}_ALL_MODELS_{scale}x_comparison.png")
            save_comparison_grid(hr_tensor, results, grid_out, cols=3)
            print(f"\nSaved Grid: {grid_out}", flush=True)
        except Exception as e:
            print(f"[WARN] Grid failed: {e}")

    if summary_history:
        with open('final_summary_metrics.csv', 'w', newline='') as f:
            fieldnames = ['Model', 'Scale', 'PSNR', 'SSIM', 'LPIPS', 'Score', 'Time(s)']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_history:
                row.update({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
                writer.writerow(row)
        print("\nDone. Summary saved to final_summary_metrics.csv")
