import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def get_mgrid(height, width):
    """Generates flattened grid of (x,y) coordinates in [-1,1]"""
    y = torch.linspace(-1, 1, steps=height)
    x = torch.linspace(-1, 1, steps=width)
    mgrid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    return mgrid.reshape(-1, 2)

def compute_metrics(hr_tensor, pred_rgb, lpips_fn, device):
    # ------------------ PSNR / SSIM (full-res) ------------------
    hr_np = hr_tensor.permute(1, 2, 0).detach().cpu().numpy()
    psnr_val = psnr(hr_np, pred_rgb, data_range=1.0)
    ssim_val = ssim(hr_np, pred_rgb, channel_axis=-1, data_range=1.0)

    # ------------------ LPIPS (CPU-safe) ------------------
    def _to_lpips_range(img_chw):
        return (img_chw.unsqueeze(0) * 2.0 - 1.0)

    pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).float().clamp(0, 1)
    hr_t   = hr_tensor.float().clamp(0, 1)

    H, W = hr_t.shape[1:]
    try:
        lpips_fn = lpips_fn.to('cpu').eval()

        if max(H, W) <= 1024:
            with torch.no_grad():
                lpips_val = lpips_fn(
                    _to_lpips_range(hr_t).cpu(),
                    _to_lpips_range(pred_t).cpu()
                ).item()
        else:
            patch = 256
            stride = 192
            ys = list(range(0, max(1, H - patch + 1), stride))
            xs = list(range(0, max(1, W - patch + 1), stride))
            if ys[-1] != H - patch: ys.append(max(0, H - patch))
            if xs[-1] != W - patch: xs.append(max(0, W - patch))

            vals = []
            with torch.no_grad():
                for y in ys:
                    for x in xs:
                        hr_p   = hr_t[:, y:y+patch, x:x+patch]
                        pred_p = pred_t[:, y:y+patch, x:x+patch]
                        # Pad if necessary
                        if hr_p.shape[1] != patch or hr_p.shape[2] != patch:
                            pad_h, pad_w = patch - hr_p.shape[1], patch - hr_p.shape[2]
                            pad = (0, max(0, pad_w), 0, max(0, pad_h))
                            hr_p = F.pad(hr_p, pad, mode='reflect')
                            pred_p = F.pad(pred_p, pad, mode='reflect')

                        v = lpips_fn(_to_lpips_range(hr_p).cpu(), _to_lpips_range(pred_p).cpu()).item()
                        vals.append(v)

            lpips_val = float(np.mean(vals)) if len(vals) else float('nan')
    except Exception:
        lpips_val = float('nan')

    return psnr_val, ssim_val, lpips_val

def _tensor_chw_to_numpy(img_chw):
    return img_chw.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

def save_rgb_image(pred_rgb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (np.clip(pred_rgb, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)

def save_tensor_image(img_chw, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (_tensor_chw_to_numpy(img_chw) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)

def save_comparison_grid(hr_tensor, results_dict, out_path, cols=3):
    import math
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panels = [("Original HR", _tensor_chw_to_numpy(hr_tensor))]
    for name, d in results_dict.items():
        panels.append((f"{name.upper()} PSNR {d['psnr']:.2f} SSIM {d['ssim']:.3f} LPIPS {d['lpips']:.3f}",
                       np.clip(d['pred'], 0, 1)))
    
    n = len(panels)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(5*cols, 5*rows))
    for i, (title, img) in enumerate(panels, 1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for j in range(n+1, rows*cols + 1):
        ax = plt.subplot(rows, cols, j); ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
