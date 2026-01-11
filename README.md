# Implicit Neural Representations (INR) for Super-Resolution and 3D Occupancy

This repository contains an experimental framework for **Implicit Neural Representations (INRs)**:
coordinate-based MLPs that represent signals as continuous functions rather than discrete grids.

It supports **two pipelines**:

- **Part 1 — Image Super-Resolution (SISR)**: learn a continuous image function from a low-resolution image and query it at arbitrary resolution.
- **Part 2 — 3D Occupancy Reconstruction**: learn a continuous occupancy function `f(x, y, z) -> {0, 1}` from sampled 3D points.

---

## Repository overview

Top-level entry points:

- `main.py` — **2D Super-Resolution** experiments
- `download_div2k.py` — helper to download/prepare DIV2K (if you use DIV2K)
- `generate_data.py` — generate a 3D occupancy dataset (`.npz`) from an `.obj` mesh
- `train_3d.py`, `train_3dv2.py` — **3D occupancy** training scripts

Core code:

- `src/models.py` — INR architectures (SIREN / WIRE / Fourier features / MFN / etc.)
- `src/config.py` — model-specific hyperparameters
- `src/trainer.py` — training loop, scheduling, batching
- `src/utils.py` — I/O and visualization utilities

Included sample assets:

- `0788.png` — sample image for quick SR sanity-check
- `dragon.obj`, `nefertiti.obj` — sample meshes
- `dragon_dataset.npz`, `nefertiti_dataset.npz` — pre-generated occupancy datasets

> Tip: if you’re unsure about arguments for any script, run:
> `python <script>.py --help`

---

## Installation

### 1) Clone

```bash
git clone https://github.com/AhmedKElshahed/INR.git
cd INR
