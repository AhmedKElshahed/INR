import os
import time
import random
import torch
import numpy as np
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
from src.utils import save_tensor_image, save_rgb_image, save_comparison_grid, save_error_map
from src.config import BEST_CONFIGS


def set_seed(seed: int):
    """Fully reproducible seeding for PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_process_image(image_path, scale):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    w = w - (w % 2)
    h = h - (h % 2)
    img = img.crop((0, 0, w, h))
    hr_tensor = transforms.ToTensor()(img)
    lr_tensor = F.interpolate(
        hr_tensor.unsqueeze(0),
        scale_factor=1 / scale,
        mode='bicubic',
        align_corners=False,
        antialias=True
    ).squeeze(0)
    return hr_tensor.clamp(0, 1), lr_tensor.clamp(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run INR Super-Resolution (multi-seed)")
    parser.add_argument("--image",  type=str, required=True, help="Path to high-res image")
    parser.add_argument("--scales", type=int, nargs='+', default=[4, 8], help="Scales to test")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="Models to run. Default: all.")
    parser.add_argument("--seeds",  type=int, default=1,
                        help="Number of random seeds (e.g. 10). Default: 1 (single run).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed batch size for reproducibility across GPU/CPU environments.
    # Larger batches (e.g. 10000 on big GPUs) reduce gradient steps per epoch,
    # causing under-training at 100 epochs and lower PSNR. 2048 ensures ~340
    # steps/epoch on a 2x-downsampled 2040x1356 image regardless of hardware.
    BATCH_SIZE = 2048
    print(f"Batch Size: {BATCH_SIZE}")

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        exit()

    img_name = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs("outputs_2d", exist_ok=True)

    try:
        lpips_fn = lpips.LPIPS(net='vgg').to('cpu').eval()
    except Exception:
        print("Warning: LPIPS failed to load.")
        lpips_fn = None

    all_models = ["siren", "fr", "wire", "finer", "mfn", "incode", "gauss", "fourier"]
    model_order = args.models if args.models else all_models
    for m in model_order:
        if m not in all_models:
            print(f"Error: Unknown model '{m}'. Available: {all_models}")
            exit()

    seed_list = list(range(args.seeds))
    multi_seed = args.seeds > 1

    DEFAULT_LR   = 1e-4
    DEFAULT_LOSS = "mse"

    # per_seed_results[model][scale] = list of metric dicts (one per seed)
    per_seed_results = {m: {s: [] for s in args.scales} for m in model_order}

    # Raw per-run CSV (one row per seed × model × scale)
    raw_csv = f'results_2d_seeds_{img_name}.csv'

    # Resume: load already-completed (model, scale, seed) from the CSV so that
    # re-running the same command after an interruption skips finished work.
    completed = set()
    if os.path.exists(raw_csv):
        with open(raw_csv, newline='') as _f:
            for _row in csv.DictReader(_f):
                completed.add((_row['model'], int(_row['scale']), int(_row['seed'])))
                # Also pre-populate per_seed_results so the final summary is correct
                _scale_int = int(_row['scale'])
                if _row['model'] in per_seed_results and _scale_int in per_seed_results[_row['model']]:
                    per_seed_results[_row['model']][_scale_int].append({
                        'PSNR':  float(_row['psnr']),
                        'SSIM':  float(_row['ssim']),
                        'LPIPS': float(_row['lpips']),
                        'MAE':   float(_row['mae']),
                        'RMSE':  float(_row['rmse']),
                        'Time':  0.0,  # not stored in CSV; use 0 for resumed runs
                    })
        if completed:
            print(f"[RESUME] {len(completed)} run(s) already done in {raw_csv} — skipping them.")

    print(f"\n{'='*70}")
    print(f"IMAGE  : {args.image}")
    print(f"SCALES : {args.scales}")
    print(f"MODELS : {model_order}")
    print(f"EPOCHS : {args.epochs}")
    print(f"SEEDS  : {seed_list}  ({'multi-seed' if multi_seed else 'single run'})")
    print(f"{'='*70}")

    for scale in args.scales:
        print(f"\n{'='*70}\nSCALE {scale}x\n{'='*70}")

        hr_tensor, lr_tensor = load_and_process_image(args.image, scale)

        # Save ground truth and LR once per scale
        save_rgb_image(hr_tensor, os.path.join("outputs_2d", f"ground_truth_{scale}x.png"))
        save_rgb_image(lr_tensor, os.path.join("outputs_2d", f"low_res_{scale}x.png"))

        results_for_grid = {}  # for comparison grid (best seed only)

        for mname in model_order:
            print(f"\n{'='*60}")
            print(f"  {mname.upper()} @ {scale}x  ({args.seeds} seed(s))")
            print(f"{'='*60}")

            for seed in seed_list:
                if (mname, scale, seed) in completed:
                    print(f"  Seed {seed}: SKIPPED (already in CSV)")
                    continue

                set_seed(seed)

                cfg = BEST_CONFIGS.get(mname, {}).copy()
                hidden_layers = cfg.pop("hidden_layers", 4)
                cfg_str = ", ".join([f"{k}={v}" for k, v in cfg.items()])

                is_first_seed = (seed == seed_list[0])
                save_ckpts = is_first_seed  # only save checkpoints for seed 0

                try:
                    model = create_model(mname, hidden_layers=hidden_layers, **cfg)
                    current_lr = 1e-3 if mname == 'gauss' else DEFAULT_LR

                    t0 = time.time()
                    pred_rgb, psnr_val, ssim_val, lpips_val, mae, rmse, error_std, error_map = train_inr_for_scale(
                        model=model,
                        hr_tensor=hr_tensor,
                        lr_tensor=lr_tensor,
                        scale=scale,
                        model_name=mname,
                        config_str=cfg_str,
                        epochs=args.epochs,
                        lr=current_lr,
                        batch_size=BATCH_SIZE,
                        device=device,
                        lpips_fn=lpips_fn,
                        loss_type=DEFAULT_LOSS,
                        checkpoint_dir=f'checkpoints_{mname}_{scale}x',
                        csv_path=raw_csv,
                        seed=seed,
                        save_checkpoints=save_ckpts,
                    )
                    duration = time.time() - t0

                    per_seed_results[mname][scale].append({
                        'PSNR': psnr_val, 'SSIM': ssim_val, 'LPIPS': lpips_val,
                        'MAE': mae, 'RMSE': rmse, 'Time': duration,
                    })

                    # Save visual outputs for seed 0 only
                    if is_first_seed:
                        save_rgb_image(pred_rgb,
                                       os.path.join("outputs_2d", f"{mname}_{scale}x.png"))
                        save_error_map(error_map,
                                       os.path.join("outputs_2d", f"{mname}_{scale}x_error.png"))
                        results_for_grid[mname] = {
                            'pred': pred_rgb, 'psnr': psnr_val,
                            'ssim': ssim_val, 'lpips': lpips_val,
                        }

                    print(f"  Seed {seed}: PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}  ({duration:.1f}s)")

                except Exception as e:
                    print(f"  [ERROR] {mname} @ {scale}x seed={seed}: {e}")
                    traceback.print_exc()
                finally:
                    try:
                        del model, pred_rgb
                    except Exception:
                        pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Save comparison grid (seed 0 results)
        if results_for_grid:
            try:
                grid_out = os.path.join("outputs_2d", f"{img_name}_grid_{scale}x.png")
                save_comparison_grid(hr_tensor, results_for_grid, grid_out, cols=3)
            except Exception as e:
                print(f"[WARN] Grid failed: {e}")

    # -------------------------------------------------------------------------
    # Aggregate: compute mean ± std across seeds for each (model, scale)
    # -------------------------------------------------------------------------
    summary_rows = []
    for mname in model_order:
        for scale in args.scales:
            runs = per_seed_results[mname][scale]
            if not runs:
                continue
            n = len(runs)

            def _stats(key):
                vals = [r[key] for r in runs]
                return float(np.mean(vals)), float(np.std(vals, ddof=0))

            psnr_m,  psnr_s  = _stats('PSNR')
            ssim_m,  ssim_s  = _stats('SSIM')
            lpips_m, lpips_s = _stats('LPIPS')
            mae_m,   mae_s   = _stats('MAE')
            time_m,  time_s  = _stats('Time')

            summary_rows.append({
                'Model': mname, 'Scale': scale, 'N_seeds': n,
                'PSNR_mean': psnr_m,   'PSNR_std':  psnr_s,
                'SSIM_mean': ssim_m,   'SSIM_std':  ssim_s,
                'LPIPS_mean': lpips_m, 'LPIPS_std': lpips_s,
                'MAE_mean':  mae_m,    'MAE_std':   mae_s,
                'Time_mean': time_m,   'Time_std':  time_s,
            })

    # Save aggregated summary
    summary_csv = f'final_summary_2d_seeds_{img_name}.csv'
    with open(summary_csv, 'w', newline='') as f:
        fields = ['Model', 'Scale', 'N_seeds',
                  'PSNR_mean', 'PSNR_std', 'SSIM_mean', 'SSIM_std',
                  'LPIPS_mean', 'LPIPS_std', 'MAE_mean', 'MAE_std',
                  'Time_mean', 'Time_std']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    # Pretty-print summary table
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY  ({args.seeds} seed(s))  —  {img_name}")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Scale':>5}  {'PSNR (mean±std)':>18}  {'SSIM (mean±std)':>18}  {'LPIPS (mean±std)':>19}")
    print(f"{'-'*80}")
    for row in summary_rows:
        psnr_str  = f"{row['PSNR_mean']:.2f}±{row['PSNR_std']:.2f}"
        ssim_str  = f"{row['SSIM_mean']:.4f}±{row['SSIM_std']:.4f}"
        lpips_str = f"{row['LPIPS_mean']:.4f}±{row['LPIPS_std']:.4f}"
        print(f"{row['Model']:<10} {row['Scale']:>4}x  {psnr_str:>18}  {ssim_str:>18}  {lpips_str:>19}")

    print(f"\nRaw per-seed data  -> {raw_csv}")
    print(f"Aggregated summary -> {summary_csv}")
