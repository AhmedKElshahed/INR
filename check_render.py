"""Verify that a newly exported mesh render matches the existing reference renders.

The thesis figures claim all renders share camera, lighting, and resolution. This
script checks that claim numerically before a new image is added, so a mismatched
export cannot silently enter a figure.

    python check_render.py path/to/new_render.png

Exit code 0 = match, 1 = mismatch.
"""

import os
import sys

import numpy as np
from PIL import Image

REF_DIR = os.path.join("thesis4", "Figures", "results")
TOL = {"tone": 8.0, "bbox_h_frac": 0.02, "aspect": 0.05}


def measure(path):
    """Return resolution, mean mesh tone, and bust bounding box of one render."""
    im = Image.open(path)
    a = np.array(im.convert("RGBA"))
    alpha = a[..., 3]

    # Renders use a transparent background; fall back to luminance if opaque.
    mask = alpha > 10 if alpha.min() < 255 else a[..., 0] > 25
    if not mask.any():
        raise ValueError(f"{path}: no foreground pixels found")

    ys, xs = np.where(mask)
    h, w = a.shape[:2]
    bbox_h = ys.max() - ys.min() + 1
    bbox_w = xs.max() - xs.min() + 1
    return {
        "size": (w, h),
        "tone": float(a[..., 0][mask].mean()),
        "bbox": (int(bbox_w), int(bbox_h)),
        "bbox_h_frac": bbox_h / h,
        "aspect": bbox_w / bbox_h,
    }


def reference_profile():
    """Average the measurements of every render already used in the thesis."""
    # The reference look is defined by the eight completed renders plus ground
    # truth. Anything Fourier-related is the render under question (the sigma=10
    # failure, or a not-yet-accepted re-export) and must not define the profile.
    refs = [
        f for f in sorted(os.listdir(REF_DIR))
        if f.endswith(".png") and "fourier" not in f.lower()
    ]
    if not refs:
        raise SystemExit(f"no reference renders found in {REF_DIR}")

    stats = [measure(os.path.join(REF_DIR, f)) for f in refs]
    sizes = {s["size"] for s in stats}
    if len(sizes) > 1:
        print(f"[warn] references disagree on resolution: {sizes}")

    return refs, {
        "size": stats[0]["size"],
        "tone": float(np.mean([s["tone"] for s in stats])),
        "bbox_h_frac": float(np.mean([s["bbox_h_frac"] for s in stats])),
        "aspect": float(np.mean([s["aspect"] for s in stats])),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    candidate = sys.argv[1]

    refs, ref = reference_profile()
    got = measure(candidate)

    print(f"reference profile from {len(refs)} renders:")
    print(f"  resolution   {ref['size'][0]}x{ref['size'][1]}")
    print(f"  mesh tone    {ref['tone']:.1f}")
    print(f"  bust height  {100 * ref['bbox_h_frac']:.1f}% of frame")
    print(f"  bust aspect  {ref['aspect']:.3f}\n")

    checks = [
        ("resolution", got["size"] == ref["size"],
         f"{got['size'][0]}x{got['size'][1]}", f"{ref['size'][0]}x{ref['size'][1]}"),
        ("mesh tone", abs(got["tone"] - ref["tone"]) <= TOL["tone"],
         f"{got['tone']:.1f}", f"{ref['tone']:.1f} +/- {TOL['tone']}"),
        ("bust height", abs(got["bbox_h_frac"] - ref["bbox_h_frac"]) <= TOL["bbox_h_frac"],
         f"{100 * got['bbox_h_frac']:.1f}%", f"{100 * ref['bbox_h_frac']:.1f}% +/- 2pp"),
        ("bust aspect", abs(got["aspect"] - ref["aspect"]) <= TOL["aspect"],
         f"{got['aspect']:.3f}", f"{ref['aspect']:.3f} +/- {TOL['aspect']}"),
    ]

    print(f"checking {os.path.basename(candidate)}:")
    for name, ok, actual, expected in checks:
        print(f"  [{'OK ' if ok else 'BAD'}] {name:12s} got {actual:12s} expected {expected}")

    if all(ok for _, ok, _, _ in checks):
        print("\nMATCH - safe to use as Figures/results/nefertiti_fourier.png")
        return 0
    print("\nMISMATCH - do not add to the thesis figures yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
