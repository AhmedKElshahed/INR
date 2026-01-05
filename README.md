# Implicit Neural Representations (INR) for Super-Resolution and 3D Occupancy

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the implementation and experimental framework for **Implicit Neural Representations (INRs)**. This work explores how coordinate-based Multi-Layer Perceptrons (MLPs) can represent continuous signals, effectively overcoming the limitations of discrete pixel grids and voxel arrays.



## Research Context

This project supports two distinct academic milestones:

1.  **Practical Work Report**: Focused on **Single-Image Super-Resolution (SISR)**. It evaluates 8 state-of-the-art INR architectures (SIREN, WIRE, FINER, etc.) across various scales ($2\times$ to $16\times$) to analyze spectral bias and high-frequency reconstruction quality.
2.  **Bachelor's Thesis**: Extends the 2D findings into **3D Occupancy**. It investigates how INRs can represent 3D volumes and occupancy grids in a memory-efficient, resolution-independent manner for robotics and computer vision applications.



---

## Key Features

* **Comprehensive Model Library**: Standardized implementations of:
    * **Activation-based**: SIREN, WIRE, GAUSS, FINER.
    * **Encoding-based**: Fourier Features.
    * **Structure-optimized**: INCODE, MFN, FR.
* **Scale-Agnostic Super-Resolution**: Ability to query the continuous function at any resolution, providing smooth transitions between scales.
* **Automated Evaluation Pipeline**: Integrated PSNR, SSIM, and LPIPS metrics using the LPIPS VGG-net backbone.
* **High-Resolution Output**: Generates clean, individual PNG reconstructions for each model and unified comparison grids for qualitative analysis.

---

## Repository Structure

```text
├── src/
│   ├── models.py      # Architecture definitions for SIREN, WIRE, FR, etc.
│   ├── trainer.py     # Training logic with dynamic batch sizing and LR scheduling
│   ├── config.py      # Best-performing hyperparameters for each architecture
│   └── utils.py       # Image processing, tensor-to-RGB saving, and grid generation
├── outputs_2d/        # Results folder containing PNG reconstructions and grids
├── main.py            # Primary script for running 2D Super-Resolution experiments
