# PartialObs-PDEBench: Partial-Observation PDE Reconstruction Benchmark (Baselines)

PartialObs-PDEBench is a **config-driven benchmark harness** for reconstructing PDE solution fields from **partial observations** (sparse sensors / subsampling / missing blocks).

This repository is designed to be:
- **Public + academic**: clean structure, explicit experiment configs, deterministic seeds, and reproducible outputs.
- **Reusable**: add PDEs, masks, and methods via small isolated modules.
- **Dataset-compatible**: supports DiffusionPDE-style `.npy` datasets (downloaded by the user; not redistributed here).

**Project page:** `https://ru1ch3n.github.io/PartialObs-PDEBench/`  
**Personal page:** `https://ru1ch3n.github.io/`

---

## Problem setting

We treat the solution field `u` as partially observed:
\[
y = M(u) + \epsilon
\]
where:
- `M` is a partial observation operator (mask/sensor pattern),
- `ε` is optional noise.

**Goal:** reconstruct the full field `u` from `(y, M)`.

---

## PDE suite

This benchmark targets four canonical PDE families:

- **Burgers (1D, time-dependent):** \(u(x,t)\)
- **Darcy (2D, elliptic):** \( -\nabla\cdot(a(x)\nabla u) = f \)
- **Poisson (2D, elliptic):** \( -\Delta u = f \)
- **Navier–Stokes (2D vorticity):** \( \partial_t \omega + u\cdot\nabla\omega = \nu \Delta\omega + f \)

---

## Partial-observation masks (M1–M3)

We standardize three observation operators:

- **M1: random points** — sample random sensor locations at a target observation ratio
- **M2: regular subsampling** — observe a strided lattice (every *k*-th point)
- **M3: block missing** — hide one or more contiguous rectangular regions (inpainting-style)

Mask configs live in `configs/mask/`.

---

## Baselines included

### Supervised reconstruction models (train yourself)
These models are trained on paired data (partial observation → full field):

- **U-Net**: a strong CNN encoder–decoder baseline with skip connections.  
  Reference: Ronneberger et al., *U-Net* (2015).  
  Paper: `https://arxiv.org/abs/1505.04597`

- **FNO (Fourier Neural Operator)**: operator learning with global Fourier-domain convolutions.  
  Reference: Li et al., *FNO* (ICLR 2021 / arXiv:2010.08895).  
  Paper: `https://arxiv.org/abs/2010.08895`  
  Official ecosystem: `https://github.com/neuraloperator/neuraloperator`

- **CNO (Convolutional Neural Operator)**: convolutional operator architecture for learning mappings between function spaces.  
  Reference: Raonić et al., *Convolutional Neural Operators* (NeurIPS 2023 / arXiv:2302.01178).  
  Paper: `https://arxiv.org/abs/2302.01178`  
  Official repo: `https://github.com/camlab-ethz/ConvolutionalNeuralOperator`

- **DeepONet**: branch/trunk operator network for learning nonlinear operators.  
  Reference: Lu et al., *DeepONet* (Nature Machine Intelligence, 2021).  
  Paper: `https://www.nature.com/articles/s42256-021-00302-5`  
  arXiv: `https://arxiv.org/abs/1910.03193`  
  Reference repo: `https://github.com/lululxvi/deeponet`

### Physics-based optimization baseline (per-instance)
- **PINN (Physics-Informed Neural Network)**: optimize network parameters per test instance using PDE residual + observation loss (and optional boundary/initial constraints).  
  Reference: Raissi et al., *PINNs* (JCP 2019).  
  Paper (DOI): `https://doi.org/10.1016/j.jcp.2018.10.045`  
  Reference repo: `https://github.com/maziarraissi/PINNs`

### Pretrained diffusion baseline (inference-only)
- **DiffusionPDE**: pretrained diffusion prior for PDE solving under partial observation; evaluated via upstream inference scripts and released checkpoints.  
  Paper: `https://arxiv.org/abs/2406.17763`  
  Repo: `https://github.com/jhhuangchloe/DiffusionPDE`

> Notes:
> - This repo is a **benchmark harness**, not an “official” implementation of the above methods.
> - Third-party datasets / weights / code are not redistributed here. See `THIRD_PARTY_NOTICES.md`.

---

# Project homepage (GitHub Pages)

A single-page project website is included in `docs/` and is designed to “fit on one page”:
- PDE definitions
- Masks (M1–M3)
- Method summaries (U-Net, FNO, CNO, DeepONet, PINN, DiffusionPDE)
- Data download + classical-solver data generation notes
- Quickstart commands
- References

## Local preview
```bash
python -m http.server 8000 --directory docs
```
Open:
```text
http://localhost:8000
```

## Publish with GitHub Pages
1) GitHub repo → **Settings** → **Pages**  
2) Build and deployment → **Deploy from a branch**  
3) Branch: `main`  
4) Folder: `/docs`  

Your site:
```text
https://ru1ch3n.github.io/PartialObs-PDEBench/
```

---

# License

- This repository’s original code is released under the **MIT License** (see `LICENSE`).
- External code/data/models are governed by upstream licenses/terms (see `THIRD_PARTY_NOTICES.md`).
