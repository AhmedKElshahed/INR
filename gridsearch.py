import os
import time
import torch
import csv
import argparse
import traceback
import gc
import itertools
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import lpips
from src.models import create_model
from src.trainer import train_inr_for_scale
from src.utils import save_rgb_image, save_error_map
from src.config import BEST_CONFIGS, GRID_SEARCH_SPACES


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


def generate_param_combos(search_space):
    """Generate all combinations from a search space dict.

    Example:
        {'a': [1, 2], 'b': [3, 4]}
        -> [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    keys = list(search_space.keys())
    values = list(search_space.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search over INR hyperparameters for 2D Super-Resolution")
    parser.add_argument("--image", type=str, required=True, help="Path to high-res image")
    parser.add_argument("--scales", type=int, nargs='+', default=[4, 8], help="Scales to test")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per run")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="Models to search (e.g. --models siren wire). Default: all models.")
    parser.add_argument("--output_dir", type=str, default="gridsearch_outputs", help="Output directory")
    parser.add_argument("--csv", type=str, default="gridsearch_results.csv", help="CSV results file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 2000
    if device.type == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        if gpu_mem > 16000:
            BATCH_SIZE = 10000
        elif gpu_mem > 8000:
            BATCH_SIZE = 5000
    print(f"Batch Size: {BATCH_SIZE}")

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        exit()

    img_name = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        lpips_fn = lpips.LPIPS(net='vgg').to('cpu').eval()
    except Exception:
        print("Warning: LPIPS failed to load.")
        lpips_fn = None

    all_models = list(GRID_SEARCH_SPACES.keys())
    models_to_search = args.models if args.models else all_models

    # Validate model names
    for m in models_to_search:
        if m not in GRID_SEARCH_SPACES:
            print(f"Error: Model '{m}' not found in GRID_SEARCH_SPACES. Available: {all_models}")
            exit()

    DEFAULT_LOSS = "mse"

    # Count total runs
    total_runs = 0
    for mname in models_to_search:
        combos = generate_param_combos(GRID_SEARCH_SPACES[mname])
        total_runs += len(combos) * len(args.scales)
    print(f"\nTotal grid search runs: {total_runs}")
    print(f"Models: {models_to_search}")
    print(f"Scales: {args.scales}")

    # Collect all possible hyperparameter names across selected models
    all_hp_names = set()
    for mname in models_to_search:
        all_hp_names.update(GRID_SEARCH_SPACES[mname].keys())
    all_hp_names.discard('hidden_layers')  # handled separately
    all_hp_names = sorted(all_hp_names)

    # CSV header: fixed columns + one column per hyperparameter
    csv_fields = (
        ['image', 'model', 'scale', 'hidden_layers']
        + all_hp_names
        + ['psnr', 'ssim', 'lpips', 'mae', 'rmse', 'error_std', 'score', 'time_s']
    )
    csv_exists = os.path.exists(args.csv)
    if not csv_exists:
        with open(args.csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    run_idx = 0

    for mname in models_to_search:
        search_space = GRID_SEARCH_SPACES[mname]
        combos = generate_param_combos(search_space)
        print(f"\n{'='*70}")
        print(f"MODEL: {mname.upper()} — {len(combos)} configurations")
        print(f"{'='*70}")

        for combo in combos:
            cfg = combo.copy()
            hidden_layers = cfg.pop("hidden_layers", 4)
            cfg_str = ", ".join([f"{k}={v}" for k, v in cfg.items()])

            for scale in args.scales:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {mname.upper()} @ {scale}x | {cfg_str} | layers={hidden_layers}")

                model = None
                pred_rgb = None
                try:
                    hr_tensor, lr_tensor = load_and_process_image(args.image, scale)
                    model = create_model(mname, hidden_layers=hidden_layers, **cfg)

                    current_lr = 1e-3 if mname == 'gauss' else 1e-4

                    start_t = time.time()
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
                        checkpoint_dir=os.path.join(args.output_dir, f'ckpt_{mname}_{scale}x'),
                        csv_path=os.devnull  # suppress trainer's internal CSV; we write our own below
                    )
                    duration = time.time() - start_t
                    score = psnr_val + 10 * ssim_val - 5 * lpips_val

                    # Save prediction and error map
                    safe_cfg = cfg_str.replace(", ", "_").replace("=", "")
                    pred_path = os.path.join(args.output_dir, f"{mname}_{scale}x_{safe_cfg}.png")
                    err_path = os.path.join(args.output_dir, f"{mname}_{scale}x_{safe_cfg}_error.png")
                    save_rgb_image(pred_rgb, pred_path)
                    save_error_map(error_map, err_path)

                    # Append to CSV — one column per hyperparameter
                    row = {
                        'image': img_name, 'model': mname, 'scale': scale,
                        'hidden_layers': hidden_layers,
                        'psnr': f"{psnr_val:.2f}", 'ssim': f"{ssim_val:.4f}",
                        'lpips': f"{lpips_val:.4f}", 'mae': f"{mae:.4f}",
                        'rmse': f"{rmse:.4f}", 'error_std': f"{error_std:.4f}",
                        'score': f"{score:.2f}", 'time_s': f"{duration:.1f}",
                    }
                    # Fill in the hyperparameter columns for this model
                    for hp_name in all_hp_names:
                        row[hp_name] = cfg.get(hp_name, '')
                    with open(args.csv, 'a', newline='') as f:
                        csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

                except Exception as e:
                    print(f"[ERROR] {mname} @ {scale}x failed: {e}")
                    traceback.print_exc()
                finally:
                    del pred_rgb, model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Print summary of best configs per model
    print(f"\n{'='*70}")
    print("GRID SEARCH COMPLETE — Best configs per model:")
    print(f"{'='*70}")
    try:
        import pandas as pd
        df = pd.read_csv(args.csv)
        for mname in models_to_search:
            mdf = df[df['model'] == mname]
            if mdf.empty:
                continue
            best = mdf.loc[mdf['psnr'].astype(float).idxmax()]
            print(f"\n{mname.upper()}:")
            print(f"  Config: {best['config']}")
            print(f"  Scale:  {best['scale']}x")
            print(f"  PSNR:   {best['psnr']} | SSIM: {best['ssim']} | LPIPS: {best['lpips']}")
            print(f"  MAE:    {best['mae']} | RMSE: {best['rmse']} | Error Std: {best['error_std']}")
    except ImportError:
        print(f"Results saved to {args.csv} (install pandas for summary table)")
    except Exception as e:
        print(f"Could not print summary: {e}")
        print(f"Results saved to {args.csv}")
