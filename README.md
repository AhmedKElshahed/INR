# Implicit Neural Representations (INR) for Super-Resolution and 3D Occupancy

This repository contains an experimental framework for **Implicit Neural Representations (INRs)**:
coordinate-based MLPs that represent signals as continuous functions rather than discrete grids.

It supports **two pipelines**:

- **Part 1 — Image Super-Resolution**: learn a continuous image function from a low-resolution image and query it at arbitrary resolution.
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

---

## Installation

### 1) Clone

```bash
git clone https://github.com/AhmedKElshahed/INR.git
cd INR
```

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell
```

### 3) Install repository dependencies
```bash
pip install -r requirements.txt
```

# PART 1 — Image Super-Resolution

### Example: Run Super-Resolution

You can specify the input image, number of training epochs, and multiple upscale factors:

```bash
python main.py --image 0788.png --epochs 100 --scales 2 4 8 16
```


# PART 2 — 3D Occupancy Reconstruction

## Step 1 — Generate Occupancy Dataset from a Mesh

To generate a training dataset from a mesh (e.g., `nefertiti.obj`), run:

```bash
python generate_data.py --mesh nefertiti.obj
```

## Step 2 — Train the Occupancy INR Model

Once the dataset is generated, you can train the 3D INR model using:

```bash
python train_3dv2.py --mesh nefertiti.obj --epochs 50
```


