#!/usr/bin/env python3
"""Static-site generator for the *docs/* pages.

Source of truth
--------------
The generator prefers per-paper JSON files under ``data/curations/*.json``.
If that folder is missing/empty, it falls back to the legacy
``scripts/research_db.ndjson`` file.

It writes:
  - docs/index.html                     (homepage: summary + paper tree)
  - docs/research/index.html            (research hub + category browser)
  - docs/research/<slug>/index.html     (one page per paper)
  - docs/pde-problems/index.html        (PDE-centric index)
  - docs/baselines/index.html           (baseline-centric index)
  - docs/builder/index.html             (interactive benchmark command builder)
  - docs/progress/index.html            (full-dataset generation/QC status)
  - docs/assets/progress.json           (machine-readable status snapshot)
  - docs/server/index.html              (Linux + SeaWulf run guide)
  - docs/contribute/index.html          (how to add/curate papers)

This repo uses GitHub Pages with /docs as the site root.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# JSON paper database (preferred)
CURATIONS_JSON_DIR = REPO_ROOT / "data" / "curations"


FULL_DATASET_PROGRESS: Dict[str, Any] = {
    "schema_version": "pdeobs-progress-v1",
    "updated_at": "2026-08-19T03:36:45-04:00",
    "campaign": "numerics-full-t15-6c7c7e31",
    "status": "final_strict_qc_running",
    "status_label": "Final strict QC running",
    "generation": {
        "complete": True,
        "shards_complete": 3360,
        "shards_expected": 3360,
        "samples_complete": 560000,
        "samples_expected": 560000,
        "percentage": 100.0,
        "pde_families": 7,
        "samples_per_pde": 80000,
        "stored_time_steps": 15,
        "storage_gib_approx": 235,
    },
    "artifacts": {
        "hdf5_complete": 3360,
        "manifest_complete": 3360,
        "sha256_complete": 3360,
        "metadata_complete": 3360,
        "quality_complete": 3360,
        "partials": 0,
        "locks": 0,
        "missing_or_mismatched": 0,
        "live_generation_jobs": 0,
        "duplicate_or_overlapping_outputs": 0,
    },
    "quality": {
        "gate": 0.05,
        "max_pde_loss": 0.049914687843795082,
        "all_shard_checks_pass": True,
        "require_all_pdes": True,
        "r4_max_pde_loss": 0.048199753441713344,
    },
    "plans": {
        "original_combined": {
            "name": "numerics-full-t15-6c7c7e31.quality-recovered-combined.jsonl",
            "sha256": "f97bf441c211cc5d8bdf9602837e826bbd360d9656b53d4e1942092f4e4d2e19",
            "unchanged": True,
        },
        "final_qc": {
            "name": "numerics-full-t15-6c7c7e31.quality-recovered-final-qc.jsonl",
            "sha256": "1173212693fe72d098c5077c73bc2314df8012c886a8e4fb78cb99d8010cfd81",
            "rows": 3360,
            "samples": 560000,
            "content_spec_mismatches": 0,
        },
    },
    "final_qc": {
        "job_id": 2130280,
        "state": "RUNNING",
        "partition": "long-40core-shared",
        "resources": "1 CPU / 8 GiB",
        "strict": True,
        "max_pde_loss": 0.05,
        "require_all_pdes": True,
        "summary_created": False,
        "previous_job": {
            "job_id": 2130270,
            "state": "FAILED",
            "exit_code": "2:0",
            "cause": "Two stale expected-plan time_steps values; artifact quality passed.",
        },
    },
}


PAPER_TREE_ASCII = r"""
AI4PDE / SciML (selected milestones)
├─ Physics-informed optimization
│  ├─ Deep Ritz (2018)
│  ├─ DGM / Deep Galerkin (2018)
│  ├─ DeepBSDE (2018)
│  └─ PINNs (2019)
│     ├─ cPINNs (2020)
│     ├─ SA-PINNs (2020)
│     ├─ XPINNs (2021)
│     ├─ gPINNs (2021)
│     └─ FBPINNs (2021)
├─ Operator learning
│  ├─ FNO (2020)
│  ├─ GKN (2020)
│  ├─ MGNO (2020)
│  ├─ DeepONet (2021)
│  ├─ PINO (2021)
│  ├─ Galerkin Transformer (2021)
│  ├─ U-NO (2022)
│  ├─ WNO (2022)
│  ├─ CNO (2023)
│  └─ U-WNO (2024)
├─ Diffusion / generative PDE inference
│  ├─ Conditional diffusion protocols (2024)
│  ├─ DiffusionPDE (2024)
│  ├─ FunDPS (2025)
│  ├─ PRISMA (2025)
│  └─ VideoPDE (2025)
├─ Graph / mesh simulators
│  ├─ GNS (ICML 2020)
│  └─ MeshGraphNets (ICLR 2021)
└─ Benchmarks and datasets
   ├─ PDEBench (2022)
   ├─ PDEArena (2022)
   ├─ FourCastNet (2022)
   └─ GraphCast (2023)
""".strip("\n")

PAPER_TREE_ASCII_LINKED = r"""
AI4PDE / SciML (selected milestones)
├─ <a href="research/?q=physics">Physics-informed optimization</a>
│  ├─ <a href="research/paper/?slug=deep-ritz">Deep Ritz (2018)</a>
│  ├─ <a href="research/paper/?slug=dgm">DGM / Deep Galerkin (2018)</a>
│  ├─ <a href="research/paper/?slug=deepbsde">DeepBSDE (2018)</a>
│  └─ <a href="research/paper/?slug=pinn">PINNs (2019)</a>
│     ├─ <a href="research/paper/?slug=cpinn">cPINNs (2020)</a>
│     ├─ <a href="research/paper/?slug=sa-pinn">SA-PINNs (2020)</a>
│     ├─ <a href="research/paper/?slug=xpinn">XPINNs (2021)</a>
│     ├─ <a href="research/paper/?slug=gpinn">gPINNs (2021)</a>
│     └─ <a href="research/paper/?slug=fbpinns">FBPINNs (2021)</a>
├─ <a href="research/?method=Operator%20learning">Operator learning</a>
│  ├─ <a href="research/paper/?slug=fno">FNO (2020)</a>
│  ├─ <a href="research/paper/?slug=gkn">GKN (2020)</a>
│  ├─ <a href="research/paper/?slug=mgno">MGNO (2020)</a>
│  ├─ <a href="research/paper/?slug=deeponet">DeepONet (2021)</a>
│  ├─ <a href="research/paper/?slug=pino">PINO (2021)</a>
│  ├─ <a href="research/paper/?slug=galerkin-transformer">Galerkin Transformer (2021)</a>
│  ├─ <a href="research/paper/?slug=u-no">U-NO (2022)</a>
│  ├─ <a href="research/paper/?slug=wno">WNO (2022)</a>
│  ├─ <a href="research/paper/?slug=cno">CNO (2023)</a>
│  └─ <a href="research/paper/?slug=u-wno">U-WNO (2024)</a>
├─ <a href="research/?method=Diffusion">Diffusion / generative PDE inference</a>
│  ├─ <a href="research/paper/?slug=conditional-diffusion-pde">Conditional diffusion protocols (2024)</a>
│  ├─ <a href="research/paper/?slug=diffusionpde">DiffusionPDE (2024)</a>
│  ├─ <a href="research/paper/?slug=fundps">FunDPS (2025)</a>
│  ├─ <a href="research/paper/?slug=prisma">PRISMA (2025)</a>
│  └─ <a href="research/paper/?slug=videopde">VideoPDE (2025)</a>
├─ <a href="research/?method=Graph%20%2F%20mesh">Graph / mesh simulators</a>
│  ├─ <a href="research/paper/?slug=gns">GNS (ICML 2020)</a>
│  └─ <a href="research/paper/?slug=meshgraphnets">MeshGraphNets (ICLR 2021)</a>
└─ <a href="research/?method=Benchmark">Benchmarks and datasets</a>
   ├─ <a href="research/paper/?slug=pdebench">PDEBench (2022)</a>
   ├─ <a href="research/paper/?slug=pdearena">PDEArena (2022)</a>
   ├─ <a href="research/paper/?slug=fourcastnet">FourCastNet (2022)</a>
   └─ <a href="research/paper/?slug=graphcast">GraphCast (2023)</a>
""".strip("\n")

PAPER_TREE_MERMAID = r"""
flowchart TD
  Root["AI4PDE / SciML (selected milestones)"]

  %% Physics-informed optimization (PINN family)
  Root --> PI["Physics-informed optimization"]
  PI --> DeepRitz["Deep Ritz (2018)"]
  PI --> DGM["DGM / Deep Galerkin (2018)"]
  PI --> DeepBSDE["DeepBSDE (2018)"]
  PI --> PINN["PINNs (2019)"]
  PINN --> cPINN["cPINNs (2020)"]
  PINN --> SAPINN["SA-PINNs (2020)"]
  PINN --> XPINN["XPINNs (2021)"]
  PINN --> gPINN["gPINNs (2021)"]
  PINN --> FBPINN["FBPINNs (2021)"]

  %% Operator learning (neural operators)
  Root --> OL["Operator learning"]
  OL --> DeepONet["DeepONet (2021)"]
  OL --> FNO["FNO (2020)"]
  FNO --> PINO["PINO (2021)"]
  FNO --> GalerkinT["Galerkin Transformer (2021)"]
  FNO --> UNO["U-NO (2022)"]
  FNO --> WNO["WNO (2022)"]
  WNO --> UWNO["U-WNO (2024)"]
  FNO --> CNO["CNO (2023)"]
  OL --> GKN["GKN (2020)"]
  OL --> MGNO["MGNO (2020)"]

  %% Diffusion / generative inference
  Root --> DiffGen["Diffusion / generative PDE inference"]
  DiffGen --> CondDiff["Conditional diffusion protocols (2024)"]
  CondDiff --> DiffPDE["DiffusionPDE (2024)"]
  DiffPDE --> FunDPS["FunDPS (2025)"]
  FunDPS --> PRISMA["PRISMA (2025)"]
  DiffPDE --> VideoPDE["VideoPDE (2025)"]

  %% Graph simulators
  Root --> GraphSim["Graph / mesh simulators"]
  GraphSim --> GNS["GNS (ICML 2020)"]
  GraphSim --> MGN["MeshGraphNets (ICLR 2021)"]

  %% Benchmarks / datasets
  Root --> Bench["Benchmarks and datasets"]
  Bench --> PDEBench["PDEBench (2022)"]
  Bench --> PDEArena["PDEArena (2022)"]
  Bench --> FourCastNet["FourCastNet (2022)"]
  FourCastNet --> GraphCast["GraphCast (2023)"]

  %% Clickable links (homepage)
  %% - Paper nodes go to curated pages.
  %% - Category nodes go to the Research tab with an initial filter.
  click Root "research/" "Open the research index" _self

  click PI "research/?method=PINN%20%2F%20physics-constrained" "Filter: PINN / physics-constrained" _self
  click DeepRitz "research/paper/?slug=deep-ritz" "Deep Ritz (2018)" _self
  click DGM "research/paper/?slug=dgm" "Deep Galerkin Method (2018)" _self
  click DeepBSDE "research/paper/?slug=deepbsde" "DeepBSDE (2018)" _self
  click PINN "research/paper/?slug=pinn" "PINNs (2019)" _self
  click cPINN "research/paper/?slug=cpinn" "cPINNs (2020)" _self
  click SAPINN "research/paper/?slug=sa-pinn" "SA-PINNs (2020)" _self
  click XPINN "research/paper/?slug=xpinn" "XPINNs (2021)" _self
  click gPINN "research/paper/?slug=gpinn" "gPINNs (2021)" _self
  click FBPINN "research/paper/?slug=fbpinns" "FBPINNs (2021)" _self

  click OL "research/?method=Operator%20learning" "Filter: Operator learning" _self
  click DeepONet "research/paper/?slug=deeponet" "DeepONet (2021)" _self
  click FNO "research/paper/?slug=fno" "Fourier Neural Operator (2020)" _self
  click PINO "research/paper/?slug=pino" "Physics-Informed Neural Operator (2021)" _self
  click GalerkinT "research/paper/?slug=galerkin-transformer" "Galerkin Transformer (2021)" _self
  click UNO "research/paper/?slug=u-no" "U-NO (2022)" _self
  click WNO "research/paper/?slug=wno" "WNO (2022)" _self
  click UWNO "research/paper/?slug=u-wno" "U-WNO (2024)" _self
  click CNO "research/paper/?slug=cno" "CNO (2023)" _self
  click GKN "research/paper/?slug=gkn" "Graph Kernel Network (2020)" _self
  click MGNO "research/paper/?slug=mgno" "MGNO (2020)" _self

  click DiffGen "research/?method=Diffusion" "Filter: Diffusion" _self
  click DiffPDE "research/paper/?slug=diffusionpde" "DiffusionPDE (2024)" _self
  click FunDPS "research/paper/?slug=fundps" "FunDPS (2025)" _self
  click PRISMA "research/paper/?slug=prisma" "PRISMA (2025)" _self
  click VideoPDE "research/paper/?slug=videopde" "VideoPDE (2025)" _self

  click GraphSim "research/?method=Graph%20%2F%20mesh" "Filter: Graph / mesh" _self
  click GNS "research/paper/?slug=gns" "GNS (ICML 2020)" _self
  click MGN "research/paper/?slug=meshgraphnets" "MeshGraphNets (ICLR 2021)" _self

  click Bench "benchmark/" "Benchmark tab" _self
  click PDEBench "research/paper/?slug=pdebench" "PDEBench (2022)" _self
  click PDEArena "research/paper/?slug=pdearena" "PDEArena (2022)" _self
  click FourCastNet "research/paper/?slug=fourcastnet" "FourCastNet (2022)" _self
  click GraphCast "research/paper/?slug=graphcast" "GraphCast (2023)" _self

  click CondDiff "research/paper/?slug=conditional-diffusion-pde" "Open paper page"

  %% Theme tweaks (dark)
  classDef cat fill:#121826,stroke:#223047,color:#e7edf5;
  classDef node fill:#0f1522,stroke:#223047,color:#e7edf5;
  class Root,PI,OL,DiffGen,GraphSim,Bench cat;
  class DeepRitz,DGM,DeepBSDE,PINN,cPINN,SAPINN,XPINN,gPINN,FBPINN,DeepONet,FNO,PINO,GalerkinT,UNO,WNO,UWNO,CNO,GKN,MGNO,CondDiff,DiffPDE,FunDPS,PRISMA,VideoPDE,GNS,MGN,PDEBench,PDEArena,FourCastNet,GraphCast node;
""".strip("\n")


AI4PDE_SDE_TREE_ASCII = r"""
AI4PDE + AI4SDE (taxonomy)
├─ Physics-informed optimization (PINN family)
├─ Operator learning (neural operators)
├─ Graph / mesh simulators
├─ Generative inference (diffusion / SDE bridges)
└─ Benchmarks and datasets
""".strip("\n")

AI4PDE_SDE_TREE_MERMAID = r"""
flowchart TD
  R["AI4PDE + AI4SDE: a taxonomy (high-level)"]

  R --> Phys["Physics-constrained learning"]
  Phys --> PINNfam["PINN-style residual minimization"]
  Phys --> Hybrid["Hybrid: data + physics losses"]

  R --> Op["Operator learning"]
  Op --> NO["Neural operators (FNO/DeepONet/...)"]
  Op --> ROM["Learned ROM / reduced models"]

  R --> Graph["Graph / mesh simulators"]
  Graph --> MP["Message passing / GNN solvers"]
  Graph --> Mesh["Mesh-based neural fields"]

  R --> Gen["Generative / probabilistic modeling"]
  Gen --> Score["Score-based / diffusion models"]
  Gen --> Bridge["Diffusion/SDE bridges (conditioning)"]
  Gen --> UQ["Uncertainty quantification"]

  R --> Theory["Theory & guarantees"]
  Theory --> Approx["Approximation / expressivity"]
  Theory --> Stability["Stability / generalization"]

  R --> Bench["Benchmarks"]

  %% Clickable links (homepage)
  click R "research/" "Open the research index" _self
  click Phys "research/?method=PINN%20%2F%20physics-constrained" "Filter: PINN / physics-constrained" _self
  click PINNfam "research/?method=PINN%20%2F%20physics-constrained" "Filter: PINN / physics-constrained" _self
  click Hybrid "research/?q=hybrid" "Search: hybrid" _self

  click Op "research/?method=Operator%20learning" "Filter: Operator learning" _self
  click NO "research/?q=neural%20operator" "Search: neural operator" _self
  click ROM "research/?q=reduced%20order" "Search: reduced order" _self

  click Graph "research/?method=Graph%20%2F%20mesh" "Filter: Graph / mesh" _self
  click MP "research/?q=message%20passing" "Search: message passing" _self
  click Mesh "research/?q=mesh" "Search: mesh" _self

  click Gen "research/?method=Diffusion" "Filter: Diffusion" _self
  click Score "research/?method=Diffusion" "Filter: Diffusion" _self
  click Bridge "research/?q=bridge" "Search: bridge" _self
  click UQ "research/?q=uncertainty" "Search: uncertainty" _self

  click Theory "research/?q=theory" "Search: theory" _self
  click Approx "research/?q=approximation" "Search: approximation" _self
  click Stability "research/?q=stability" "Search: stability" _self

  click Bench "benchmark/" "Benchmark tab" _self
""".strip("\n")


# Class-level math templates. Used when a paper doesn't provide a manually curated
# `core_math` section.
METHOD_MATH: Dict[str, List[str]] = {
    "PINN / physics-constrained": [
        r"u_\theta = \mathrm{NN}_\theta(x,t)\\qquad \mathcal{N}[u]=0\ \text{(PDE residual)}",
        r"\min_\theta\ \underbrace{\|u_\theta- u_{data}\|^2}_{\mathcal{L}_{data}} + \lambda\underbrace{\|\mathcal{N}[u_\theta]\|^2}_{\mathcal{L}_{PDE}} + \mu\underbrace{\|\mathcal{B}[u_\theta]\|^2}_{\mathcal{L}_{BC/IC}}",
    ],
    "Operator learning": [
        r"G_\theta: a(\cdot)\mapsto u(\cdot)\ \ \text{(solution operator)}",
        r"\min_\theta\ \sum_i \|G_\theta(a_i)-u_i\|^2\ \ \text{(+ optional physics / residual regularization)}",
    ],
    "Diffusion": [
        r"x_t = \alpha(t)\,x_0 + \sigma(t)\,\epsilon\ ,\ \epsilon\sim\mathcal{N}(0,I)",
        r"\min_\theta\ \mathbb{E}_{t,x_0,\epsilon}\ \|\epsilon-\epsilon_\theta(x_t,t,c)\|^2\ \ \text{(conditioning }c: \text{measurements/masks)}",
        r"\text{Sampling: iterate a reverse process so }x_0\sim p_\theta(\cdot\mid c)\ \text{matches observations + physics}",
    ],
    "Graph / mesh": [
        r"h_i^{(\ell+1)} = \phi\Big(h_i^{(\ell)},\ \sum_{j\in\mathcal{N}(i)} \psi(h_i^{(\ell)},h_j^{(\ell)},e_{ij})\Big)",
        r"\min_\theta\ \sum_t\ \|u_{t+1}-\mathrm{GNN}_\theta(u_t,\text{mesh})\|^2\ \ \text{(rollout / one-step)}",
    ],
    "Transformers": [
        r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d})V\ \ \text{(global token mixing)}",
    ],
    "Benchmark": [
        r"\text{(No single method.) Benchmarks define datasets, masks, metrics, and protocols.}",
    ],
    "SciML": [
        r"\text{A broad bucket: see the method class tags for the closest training objective.}",
    ],
}


# ---------------------------
# Auto-tagging helpers
# ---------------------------


def html_escape_pre(s: str) -> str:
    """Escape text for <pre> blocks.

    We escape '&' and '<' (HTML-sensitive), but *not* '>' so Mermaid graphs keep
    the literal '-->' tokens (some Mermaid renderers read innerHTML).
    """
    return s.replace("&", "&amp;").replace("<", "&lt;")


def _dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def normalize_pde_tag(tag: str) -> str:
    """Normalize PDE tags to keep the site consistent."""
    t = tag.strip()
    low = t.lower()
    # Canonicalize a few common variants.
    if low == "wave":
        return "Wave equation"
    if low in {"wave eq", "wave pde"}:
        return "Wave equation"
    if low in {"darcy", "darcy flow"}:
        return "Darcy flow"
    if low in {"advection-diffusion", "convection-diffusion", "advection diffusion"}:
        return "Advection(-diffusion)"
    if low in {"reaction diffusion", "reaction-diffusion"}:
        return "Reaction–diffusion"
    if low in {"kuramoto sivashinsky", "kuramoto-sivashinsky", "ks"}:
        return "Kuramoto–Sivashinsky"
    return t


def infer_pdes(p: Dict[str, Any]) -> List[str]:
    """Infer PDE/problem tags from metadata (conservative, title-based).

    Returns a *possibly empty* list. We intentionally avoid guessing specific
    PDEs when the title/category doesn't mention them.
    """
    text = " ".join(
        [
            str(p.get("full_title", "")),
            str(p.get("short_title", "")),
            str(p.get("category", "")),
            " ".join(p.get("badges", []) or []),
        ]
    )
    t = text.lower()

    tags: List[str] = []

    def has_any(*keys: str) -> bool:
        return any(k in t for k in keys)

    # --- Explicit PDE names (high precision) ---
    if has_any("navier", "stokes"):
        tags.append("Navier–Stokes")
    if "burgers" in t:
        tags.append("Burgers")
    if "darcy" in t or "permeability" in t or "porous media" in t:
        tags.append("Darcy flow")
    if "poisson" in t:
        tags.append("Poisson")
    if "helmholtz" in t:
        tags.append("Helmholtz")
    if has_any("heat equation", "heat conduction", "heat transfer") or (
        " heat " in f" {t} " and has_any("conduction", "thermal")
    ):
        tags.append("Heat")
    if has_any("advection-diffusion", "convection-diffusion", "hyperbolic-transport") or (
        "advection" in t and "diffusion" in t and "diffusion model" not in t
    ):
        tags.append("Advection(-diffusion)")
    elif "advection" in t or "transport equation" in t:
        tags.append("Advection(-diffusion)")

    if has_any("reaction-diffusion", "reaction diffusion"):
        tags.append("Reaction–diffusion")
    if has_any("allen-cahn", "allen cahn"):
        tags.append("Allen–Cahn")
    if has_any("cahn-hilliard", "cahn hilliard"):
        tags.append("Cahn–Hilliard")
    if has_any("kuramoto", "sivashinsky"):
        tags.append("Kuramoto–Sivashinsky")
    if has_any("korteweg", "kdv"):
        tags.append("Korteweg–De Vries")
    if has_any("schrödinger", "schrodinger"):
        tags.append("Schrödinger")
    if has_any("wave equation", "acoustic wave", "seismic wave"):
        tags.append("Wave equation")
    if has_any("maxwell", "electromagnet"):
        tags.append("Maxwell")
    if "shallow water" in t:
        tags.append("Shallow water")
    if has_any("black-scholes", "black scholes"):
        tags.append("Black–Scholes")
    if "biharmonic" in t:
        tags.append("Biharmonic")

    # --- Domain-to-problem tags (lower precision, but still informative) ---
    # Only add when explicit PDE names were not found.
    if not tags:
        if has_any("weather", "climate", "atmospheric", "era5", "nwp", "forecast"):
            tags.append("Atmospheric dynamics (primitive equations)")
        elif has_any("hemodynamics", "cfd", "turbulence") or (
            "fluid" in t and "fluid" not in "differential"
        ):
            tags.append("Fluid dynamics")
        elif has_any("elasticity", "solid mechanics", "fracture", "fatigue", "plasticity"):
            tags.append("Solid mechanics")

    return _dedup_keep_order([normalize_pde_tag(x) for x in tags])


def infer_tasks(p: Dict[str, Any]) -> List[str]:
    """Infer task tags from title/category.

    This is intentionally higher-level than PDE tags and aims to avoid empty
    'Tasks' fields on the Research table.
    """
    title = str(p.get("full_title", ""))
    cat = str(p.get("category", ""))
    mcls = str(p.get("method_class", "SciML"))
    text = f"{title} {cat} {mcls}".lower()

    tasks: List[str] = []

    def add(label: str, *keys: str) -> None:
        if any(k in text for k in keys):
            tasks.append(label)

    # Category-driven (lists are often organized by intent)
    add("Survey / review", "survey", "review", "tutorial")
    add("Benchmark / dataset", "benchmark", "dataset", "arena", "pdebench")
    add("Software / toolkit", "software", "library", "package", "toolbox", "toolkit", "framework")

    add(
        "Training acceleration / stabilization",
        "accelerat",
        "fast",
        "efficient",
        "speed",
        "precondition",
        "natural gradient",
        "optimizer",
        "conflict",
        "loss balancing",
        "curriculum",
        "adaptive weight",
        "kronecker",
    )
    add(
        "Theory / analysis",
        "analysis",
        "convergence",
        "error",
        "generalization",
        "bounds",
        "mismatch",
        "failure mode",
        "loss landscape",
        "theory",
    )
    add(
        "Adaptive sampling / active learning",
        "active learning",
        "adaptive sampling",
        "residual-based",
        "causal sampling",
        "importance sampling",
    )
    add(
        "Uncertainty quantification",
        "uncertainty",
        "bayesian",
        "probabilistic",
        "gaussian process",
        "uq",
    )

    add(
        "Inverse problem / reconstruction",
        "inverse",
        "reconstruction",
        "tomography",
        "data assimilation",
        "identification",
        "parameter estimation",
        "unknown coefficient",
    )
    add(
        "PDE discovery / identification",
        "discover",
        "discovery",
        "learning pdes",
        "pde-net",
        "sindy",
        "model discovery",
        "equation",
    )

    add(
        "Forward prediction / rollout",
        "forecast",
        "prediction",
        "predicting",
        "rollout",
        "simulation",
        "time-dependent",
        "long-term",
    )
    add(
        "Operator learning / surrogate modeling",
        "operator",
        "neural operator",
        "deeponet",
        "fourier neural operator",
        "fno",
    )

    add(
        "Generative reconstruction / inpainting",
        "diffusion",
        "generative",
        "inpainting",
        "sampling",
        "posterior",
    )
    add("Graph / mesh simulation", "graph", "mesh")
    add(
        "Neural ODE/SDE modeling",
        "neural ode",
        "neural odes",
        "sde",
        "stochastic differential",
        "controlled differential",
    )

    # Category keywords from upstream lists
    if "accerleration" in text or "acceleration" in text:
        tasks.append("Training acceleration / stabilization")
    if "analysis" in cat.lower():
        tasks.append("Theory / analysis")
    if "probabilistic" in cat.lower() or "uncertainty" in cat.lower():
        tasks.append("Uncertainty quantification")
    if "parallel" in cat.lower():
        tasks.append("Parallel / scalable training")
    if "meta" in cat.lower() or "transfer" in cat.lower():
        tasks.append("Transfer / meta-learning")

    # Fallbacks by method class
    if not tasks:
        if "diffusion" in mcls.lower():
            tasks.append("Generative reconstruction / inpainting")
        elif "operator" in mcls.lower():
            tasks.append("Operator learning / surrogate modeling")
        elif "pinn" in mcls.lower():
            tasks.append("Physics-informed solving (general)")
        elif "benchmark" in mcls.lower():
            tasks.append("Benchmark / dataset")
        elif "graph" in mcls.lower():
            tasks.append("Graph / mesh simulation")
        elif "transform" in mcls.lower():
            tasks.append("Surrogate modeling (transformer)")
        else:
            tasks.append("Scientific ML (general)")

    return _dedup_keep_order(tasks)


def infer_method_class(p: Dict[str, Any]) -> str:
    """Best-effort method taxonomy label.

    This is only a fallback when a paper JSON/DB entry does not specify
    ``method_class``.
    """
    title = str(p.get("full_title") or p.get("title") or "").lower()
    cat = str(p.get("category") or "").lower()
    text = f"{title} {cat}"

    if any(k in text for k in ["diffusion", "score-based", "score based", "denoising", "sde"]):
        return "Diffusion"
    if any(
        k in text
        for k in [
            "neural operator",
            "operator learning",
            "deeponet",
            "fno",
            "fourier neural operator",
        ]
    ):
        return "Operator learning"
    if any(
        k in text
        for k in [
            "pinn",
            "physics-informed",
            "physics informed",
            "physics-constrained",
            "physics constrained",
        ]
    ):
        return "PINN / physics-constrained"
    if any(k in text for k in ["graph", "mesh", "gns", "meshgraph", "mgno", "message passing"]):
        return "Graph / mesh"
    if any(k in text for k in ["transformer", "attention"]):
        return "Transformers"
    if any(k in text for k in ["benchmark", "dataset", "suite", "arena"]):
        return "Benchmark"
    return "SciML"


def _as_list(v: Any) -> List[str]:
    return v if isinstance(v, list) else []


def get_manual_list(p: Dict[str, Any], key: str) -> List[str]:
    return _as_list(p.get(key))


def get_auto_list(p: Dict[str, Any], key: str) -> List[str]:
    auto = p.get("auto") if isinstance(p.get("auto"), dict) else {}
    return _as_list(auto.get(key))


def get_display_list(p: Dict[str, Any], key: str) -> Tuple[List[str], bool]:
    """Return (list, is_auto).

    `is_auto=True` means the manual list is empty and we are showing auto-suggestions.
    """
    manual = get_manual_list(p, key)
    auto = get_auto_list(p, key)
    if manual:
        return manual, False
    return auto, bool(auto)


def load_db() -> List[Dict[str, Any]]:
    """Load the paper database.

    **Index list (JSON Lines):**
      - scripts/research_db.ndjson

    **Curations (per-paper JSON overrides):**
      - data/curations/<slug>.json

    Notes
    -----
    - We treat every paper as an *index* entry by default.
    - A paper becomes *curated* only if it has a corresponding curation JSON file
      (or that JSON explicitly sets ``status`` to ``curated``).
    """

    def _as_list(x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    def _normalize(p: Dict[str, Any], *, force_status: Optional[str] = None) -> Dict[str, Any]:
        # Title aliases
        if p.get("title") and not p.get("full_title"):
            p["full_title"] = p["title"]
        if p.get("full_title") and not p.get("title"):
            p["title"] = p["full_title"]

        # Required: slug
        slug = (p.get("slug") or "").strip()
        if not slug:
            return {}

        p["slug"] = slug

        # Defaults
        p["links"] = p.get("links") or {}
        p["badges"] = _as_list(p.get("badges"))
        p["pdes"] = _as_list(p.get("pdes"))
        p["tasks"] = _as_list(p.get("tasks"))

        # Ensure auto field exists for later inference
        auto = p.get("auto") or {}
        if not isinstance(auto, dict):
            auto = {}
        auto.setdefault("pdes", [])
        auto.setdefault("tasks", [])
        p["auto"] = auto
        # Index entries are metadata-only; treat any provided pdes/tasks as *auto* tags.
        if force_status == "index":
            seed_pdes = list(p.get("pdes") or [])
            seed_tasks = list(p.get("tasks") or [])
            p["pdes"] = []
            p["tasks"] = []
            auto["pdes"] = seed_pdes + list(auto.get("pdes") or [])
            auto["tasks"] = seed_tasks + list(auto.get("tasks") or [])
            p["auto"] = auto

        # Normalize a few optional structured fields (curated only)
        for key in [
            "contrib",
            "benefits",
            "theory",
            "setting",
            "data_setting",
            "model_setting",
            "training_setting",
            "interesting",
        ]:
            if key in p:
                p[key] = _as_list(p.get(key))

        if (
            "results_tables" in p
            and p["results_tables"] is not None
            and not isinstance(p["results_tables"], list)
        ):
            p["results_tables"] = _as_list(p["results_tables"])

        # Year type
        if isinstance(p.get("year"), str) and p["year"].strip().isdigit():
            p["year"] = int(p["year"].strip())

        # Status
        if force_status is not None:
            p["status"] = force_status
        else:
            p["status"] = p.get("status") or "index"

        return p

    # 1) Load base index list (NDJSON)
    by_slug: Dict[str, Dict[str, Any]] = {}
    ndjson_path = REPO_ROOT / "scripts" / "research_db.ndjson"
    if ndjson_path.exists():
        for line in ndjson_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            p = _normalize(obj, force_status="index")
            if not p:
                continue
            by_slug[p["slug"]] = p

    # 2) Apply curations (JSON)
    if CURATIONS_JSON_DIR.exists():
        for fp in sorted(CURATIONS_JSON_DIR.glob("*.json")):
            if fp.name.startswith("_"):
                continue
            try:
                cur = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(cur, dict):
                continue

            # Allow slug to be omitted if filename matches
            cur_slug = (cur.get("slug") or fp.stem).strip()
            if not cur_slug:
                continue
            cur["slug"] = cur_slug

            # Attach path for contribution hints
            try:
                cur["_curation_path"] = str(fp.relative_to(REPO_ROOT)).replace("\\", "/")
            except Exception:
                cur["_curation_path"] = str(fp)

            # Force curated status
            cur = _normalize(cur, force_status="curated")
            if not cur:
                continue

            base = by_slug.get(
                cur_slug,
                {
                    "slug": cur_slug,
                    "status": "index",
                    "auto": {"pdes": [], "tasks": []},
                    "links": {},
                },
            )
            base.update(cur)
            by_slug[cur_slug] = base

    return list(by_slug.values())


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def ul(items: List[str]) -> str:
    if not items:
        return ""
    lis = "\n".join(f"<li>{i}</li>" for i in items)
    return f"<ul>\n{lis}\n</ul>"


def placeholder(text: str = "Not extracted yet.") -> str:
    return f'<p class="muted">{html_escape(text)}</p>'


def ul_or_placeholder(items: List[str], text: str = "Not extracted yet.") -> str:
    return ul(items) if items else placeholder(text)


def render_math_block(lines: List[str]) -> str:
    if not lines:
        return ""
    blocks = []
    for eq in lines:
        # Render as display math for readability
        blocks.append(f"\\[{eq}\\]")
    return '<div class="equation">' + "\n".join(blocks) + "</div>"


def badges(items: List[str]) -> str:
    if not items:
        return ""
    spans = "\n".join(f'<span class="badge">{html_escape(i)}</span>' for i in items)
    return f'<div class="badges">\n{spans}\n</div>'


def nav(root: str, current: str) -> str:
    # current: one of home, progress, research, pde, baselines, benchmark, builder, run, contribute
    def a(href: str, label: str, key: str, *, primary: bool = False) -> str:
        aria = ' aria-current="page"' if key == current else ""
        css_class = ' class="nav-primary"' if primary else ""
        return f'<a href="{href}"{css_class}{aria}>{label}</a>'

    return (
        '<nav class="nav">'
        + a(f"{root}index.html", "Overview", "home")
        + a(f"{root}progress/", "Progress", "progress")
        + a(f"{root}builder/", "Benchmark Builder", "builder", primary=True)
        + a(f"{root}benchmark/", "Benchmark", "benchmark")
        + a(f"{root}server/", "Run", "run")
        + a(f"{root}research/", "Paper Library", "research")
        + a(f"{root}pde-problems/", "By PDE", "pde")
        + a(f"{root}baselines/", "By Method", "baselines")
        + a(f"{root}contribute/", "Contribute", "contribute")
        + "</nav>"
    )


def page(
    *,
    title: str,
    root: str,
    current: str,
    hero_h1: str,
    hero_subtitle_html: str,
    hero_meta_html: str = "",
    hero_card_html: str = "",
    extra_head: str = "",
    body_html: str,
) -> str:

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>{html_escape(title)}</title>
  <link rel=\"stylesheet\" href=\"{root}assets/style.css?v=2026-08-18-builder-first\" />
  {extra_head}
</head>

<body>
  <header class=\"hero\">
    <div class=\"container\">
      <div class=\"hero-top\">
        <div>
          <h1>{hero_h1}</h1>
          <p class=\"subtitle\">{hero_subtitle_html}</p>
          {hero_meta_html}
        </div>

        {hero_card_html}
      </div>

      {nav(root, current)}
    </div>
  </header>

  <main class=\"container\">
    {body_html}

    <footer class=\"footer\">
      <div class=\"muted\">Generated deterministically from repository sources.</div>
    </footer>
  </main>
</body>
</html>
"""


def paper_link(p: Dict[str, Any], root_to_docs: str) -> str:
    """Return an internal link for the paper.

    - curated papers: /research/<slug>/
    - index placeholders: /research/paper/?slug=<slug>  (single generic page)
    """
    status = str(p.get("status") or "index")
    if status == "curated":
        return f"{root_to_docs}research/{p['slug']}/"
    return f"{root_to_docs}research/paper/?slug={p['slug']}"


def render_paper_page(p: Dict[str, Any]) -> str:
    root = "../../"  # from docs/research/<slug>/index.html to docs/
    links = p.get("links", {}) or {}
    meta_lines = []
    if links.get("paper"):
        meta_lines.append(
            f'<div><b>Paper:</b> <a class="meta-link" href="{links["paper"]}" target="_blank" rel="noopener noreferrer">{html_escape(links.get("paper_label", "link"))}</a></div>'
        )
    if links.get("code"):
        meta_lines.append(
            f'<div><b>Code:</b> <a class="meta-link" href="{links["code"]}" target="_blank" rel="noopener noreferrer">repository</a></div>'
        )
    if links.get("project"):
        meta_lines.append(
            f'<div><b>Project:</b> <a class="meta-link" href="{links["project"]}" target="_blank" rel="noopener noreferrer">project page</a></div>'
        )

    meta_html = "\n".join(meta_lines)
    hero_meta = f'<div class="meta">{meta_html}</div>' if meta_lines else ""

    quick = p.get("quick_facts", [])
    hero_card = (
        '<div class="hero-card">'
        '  <div class="smallcaps">Quick facts</div>'
        f'  <p class="muted" style="margin-top:8px;">{"<br/>".join(quick)}</p>'
        f'  <p style="margin:10px 0 0;"><a href="../index.html">← Research</a> · <a href="../../index.html">Home</a></p>'
        "</div>"
    )

    # --- Page body ---
    sections: List[str] = []

    # Curation status + source path
    status = str(p.get("status", "index"))
    curation_path = p.get("_curation_path") or f"data/curations/{p.get('slug', '<slug>')}.json"
    if status != "curated":
        sections.append(
            '<div class="note">'
            "This page is currently an <b>index-only</b> placeholder. "
            "To improve it, edit the JSON file: "
            f"<code>{html_escape(str(curation_path))}</code> "
            "(see the <b>Contribute</b> tab)."
            "</div>"
        )

    tldr = (p.get("tldr") or "").strip()
    if not tldr:
        tldr_html = (
            '<p class="muted">Not curated yet. Add a 2–4 sentence TL;DR in the JSON file.</p>'
        )
    else:
        tldr_html = f"<p>{html_escape(tldr)}</p>"
    sections.append(f'<section id="tldr"><h2>TL;DR</h2>{tldr_html}</section>')

    # Problem statement (optional but strongly recommended for curated pages)
    problem = (p.get("problem") or "").strip()
    if not problem:
        problem_html = '<p class="muted">Add <code>problem:</code> to explain what the paper is trying to solve.</p>'
    else:
        problem_html = f"<p>{html_escape(problem)}</p>"
    sections.append(f'<section id="problem"><h2>Problem</h2>{problem_html}</section>')

    # Benefits vs others (optional; use bullet points)
    benefits = p.get("benefits") or p.get("advantages") or []
    if isinstance(benefits, str):
        benefits = [benefits]
    sections.append(
        '<section id="benefits"><h2>Benefits vs others</h2>'
        + ul_or_placeholder(
            benefits,
            "Add <code>benefits:</code> as a bullet list (e.g., accuracy, speed, data efficiency, stability, generalization).",
        )
        + "</section>"
    )

    # Interesting notes (optional)
    interesting = p.get("interesting") or p.get("notes") or ""
    if isinstance(interesting, list):
        interesting_html = ul_or_placeholder(
            interesting, "Add <code>interesting:</code> as bullet points."
        )
    else:
        interesting = str(interesting).strip()
        interesting_html = (
            f"<p>{html_escape(interesting)}</p>"
            if interesting
            else '<p class="muted">(Optional) Add <code>interesting:</code>.</p>'
        )
    sections.append(
        f'<section id="interesting"><h2>Interesting detail</h2>{interesting_html}</section>'
    )

    # Core method (math) + theory
    method_class = p.get("method_class", "SciML")
    math_lines = p.get("core_math", []) or METHOD_MATH.get(method_class, []) or METHOD_MATH["SciML"]
    sections.append(
        '<section id="core-math"><h2>Core method (math)</h2>'
        f'<p class="muted">Template for <b>{html_escape(method_class)}</b>. Paper-specific equations are added when manually curated.</p>'
        + (render_math_block(math_lines) if math_lines else placeholder("No template available."))
        + "</section>"
    )

    sections.append(
        '<section id="theory"><h2>Main theoretical contribution</h2>'
        + ul_or_placeholder(
            p.get("theory", []),
            "Not curated yet. Add bullet points under <code>theory</code> in JSON.",
        )
        + "</section>"
    )

    sections.append(
        '<section id="contribution"><h2>Main contribution</h2>'
        + ul_or_placeholder(
            p.get("contrib", []),
            "Not curated yet. Add bullet points under <code>contrib</code> in JSON.",
        )
        + "</section>"
    )

    # Main results (optional: quick headline summary)
    main_results = p.get("main_results") or []
    if isinstance(main_results, str):
        main_results = [main_results]

    if isinstance(main_results, list) and main_results:
        if all(isinstance(r, dict) for r in main_results):
            rows = []
            for r in main_results:
                rows.append(
                    "<tr>"
                    f"<td>{html_escape(str(r.get('metric', '')))}</td>"
                    f"<td>{html_escape(str(r.get('value', '')))}</td>"
                    f"<td>{html_escape(str(r.get('dataset', '')))}</td>"
                    f"<td>{html_escape(str(r.get('compared_to', '')))}</td>"
                    "</tr>"
                )
            main_results_html = (
                '<div class="tablewrap"><table>'
                "<thead><tr><th>Metric</th><th>Value</th><th>Dataset</th><th>Compared to</th></tr></thead>"
                "<tbody>" + "".join(rows) + "</tbody></table></div>"
            )
        else:
            main_results_html = ul_or_placeholder(
                [str(x) for x in main_results],
                "Add <code>main_results</code> as a list (either dict rows or strings).",
            )
    else:
        main_results_html = '<p class="muted">(Optional) Add <code>main_results</code> for a quick headline summary.</p>'

    sections.append(
        '<section id="main-results"><h2>Main results (headline)</h2>'
        + main_results_html
        + "</section>"
    )

    # Experiments / PDE / tasks
    pdes_display, pdes_is_auto = get_display_list(p, "pdes")
    tasks_display, tasks_is_auto = get_display_list(p, "tasks")
    pdes_title = "PDE problems" + (' <span class="muted">(auto)</span>' if pdes_is_auto else "")
    tasks_title = "Tasks" + (' <span class="muted">(auto)</span>' if tasks_is_auto else "")

    exp_html = (
        '<section id="experiments"><h2>Experiments</h2>'
        '<div class="grid2">'
        f'  <div class="card"><h3>{pdes_title}</h3>'
        + ul_or_placeholder(pdes_display, "Not specified yet.")
        + "</div>"
        f'  <div class="card"><h3>{tasks_title}</h3>'
        + ul_or_placeholder(tasks_display, "Not specified yet.")
        + "</div>"
        "</div>"
    )
    exp_html += (
        '<div class="card" style="margin-top:14px;"><h3>Experiment setting (high level)</h3>'
        + ul_or_placeholder(p.get("setting", []))
        + "</div>"
    )
    exp_html += "</section>"
    sections.append(exp_html)

    sections.append(
        '<section id="baselines"><h2>Comparable baselines</h2>'
        + ul_or_placeholder(
            p.get("baselines", []),
            "Not curated yet. Add items under <code>baselines</code> in JSON.",
        )
        + "</section>"
    )

    # Results tables
    tables = p.get("results_tables", []) or []
    if tables:
        res_parts = ['<section id="results"><h2>Main results</h2>']
        for t in tables:
            if t.get("title"):
                res_parts.append(f'<h3 class="subhead">{html_escape(t["title"])}</h3>')
            if t.get("note"):
                res_parts.append(f'<p class="muted">{t["note"]}</p>')
            header = t.get("header", [])
            rows = t.get("rows", [])
            thead = "".join(f"<th>{html_escape(h)}</th>" for h in header)
            body_rows = []
            for r in rows:
                body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            res_parts.append(
                '<div class="tablewrap"><table><thead><tr>'
                + thead
                + "</tr></thead><tbody>"
                + "\n".join(body_rows)
                + "</tbody></table></div>"
            )
        if p.get("benchmark_note"):
            res_parts.append(f'<div class="note">{p["benchmark_note"]}</div>')
        res_parts.append("</section>")
        sections.append("\n".join(res_parts))

    # Citation (BibTeX)
    links = p.get("links", {}) or {}
    bib = ""
    if isinstance(p.get("bib"), dict):
        bib = (p.get("bib") or {}).get("entry", "") or ""
    bib = (p.get("bibtex") or bib or "").strip()
    if not bib:
        key = (p.get("bibkey") or p.get("slug") or "paper").strip()
        key = re.sub(r"[^A-Za-z0-9_:-]+", "", key) or "paper"
        title = (p.get("full_title") or p.get("short_title") or "").strip()
        authors = (p.get("authors") or "").strip()
        year = str(p.get("year") or "").strip()
        venue = (p.get("venue") or "").strip()
        url = (
            links.get("paper")
            or links.get("arxiv")
            or links.get("openreview")
            or links.get("code")
            or ""
        )
        entry_type = "inproceedings" if venue else "article"
        fields = []
        if title:
            fields.append(f"  title={{ {title} }}")
        if authors:
            fields.append(f"  author={{ {authors} }}")
        if year:
            fields.append(f"  year={{ {year} }}")
        if venue:
            fields.append(f"  booktitle={{ {venue} }}")
        if url:
            fields.append(f"  url={{ {url} }}")
        bib = f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}"

    sections.append(
        '<section id="citation"><h2>Citation (BibTeX)</h2>'
        + f'<pre class="code"><code>{html_escape(bib)}</code></pre>'
        + "</section>"
    )

    body = "\n".join(sections)

    return page(
        title=f"{p['short_title']} ({p['year']}) — Research — PartialObs-PDEBench",
        root=root,
        current="research",
        hero_h1=f"{html_escape(p['short_title'])} ({p['year']})",
        hero_subtitle_html=f"<b>{html_escape(p['full_title'])}</b><br/>{html_escape(p.get('authors', ''))}",
        hero_meta_html=hero_meta + badges(p.get("badges", [])),
        hero_card_html=hero_card,
        extra_head=(
            "<script>window.MathJax={tex:{inlineMath:[['\\(','\\)'],['$','$']]}};</script>"
            '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
            '<script defer src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>'
            "<script>document.addEventListener('DOMContentLoaded',function(){if(window.mermaid){mermaid.initialize({startOnLoad:true,securityLevel:'loose',theme:'base',themeVariables:{primaryColor:'#121826',primaryTextColor:'#e7edf5',primaryBorderColor:'#223047',lineColor:'#3b4a66',secondaryColor:'#0f1522',tertiaryColor:'#0b0f14'}});}});</script>"
        ),
        body_html=body,
    )


def render_home(papers: List[Dict[str, Any]]) -> str:
    root = ""  # docs/index.html
    n_total = len(papers)
    n_curated = sum(1 for p in papers if str(p.get("status") or "index") == "curated")

    hero_card = (
        '<div class="hero-card home-builder-card">'
        '  <div class="smallcaps">Main contribution</div>'
        '  <h2>Build your partial-observation benchmark</h2>'
        '  <p class="muted">Choose PDEs, boundaries, physical regimes, observation masks, '
        "and quality gates. The Builder returns reproducible dataset YAML and run code."
        "  </p>"
        '  <a class="btn primary home-cta" href="builder/">Open Benchmark Builder</a>'
        '  <p class="home-secondary-link"><a href="benchmark/">Read the benchmark contract</a></p>'
        "</div>"
    )

    body = f"""
<section id="benchmark-first" class="section home-lead">
  <div class="smallcaps">Partial observations, controlled</div>
  <h2>One benchmark workflow from ground truth to quality report</h2>
  <p>Generate the physical trajectory first, then derive sparse observations as reproducible views. Observation masks never change the underlying ground truth.</p>
  <div class="benchmark-path" aria-label="Benchmark workflow">
    <div><span>1</span><b>Choose</b><small>PDE, boundary, setting, regime, scale</small></div>
    <div><span>2</span><b>Generate</b><small>Deterministic ground-truth trajectories</small></div>
    <div><span>3</span><b>Observe</b><small>Nine sparse or structured mask views</small></div>
    <div><span>4</span><b>Validate</b><small>Checksums, provenance, and PDE-specific losses</small></div>
  </div>
</section>

<section id="dataset-progress" class="section progress-teaser">
  <div class="section-heading-row">
    <div><div class="smallcaps">Full dataset campaign</div><h2>560,000 / 560,000 samples generated</h2></div>
    <a class="btn" href="progress/">Open full progress report</a>
  </div>
  <div class="home-metrics">
    <div class="card"><strong>3,360</strong><span>complete shards</span></div>
    <div class="card"><strong>100%</strong><span>generation complete</span></div>
    <div class="card"><strong>7</strong><span>PDE families</span></div>
    <div class="card"><strong>0</strong><span>partials or locks</span></div>
  </div>
  <div class="note"><b>Validation status:</b> every shard passes the unchanged <code>pde_loss &le; 0.05</code> gate; the final strict aggregate validator is running separately.</div>
</section>

<section id="benchmark-scope" class="section">
  <div class="section-heading-row">
    <div><div class="smallcaps">Frozen design space</div><h2>A benchmark you can inspect before running</h2></div>
    <a class="btn" href="benchmark/">Full contract</a>
  </div>
  <div class="home-metrics">
    <div class="card"><strong>7</strong><span>PDE families</span></div>
    <div class="card"><strong>4</strong><span>boundary protocols</span></div>
    <div class="card"><strong>10</strong><span>condition settings</span></div>
    <div class="card"><strong>3</strong><span>physical regimes</span></div>
    <div class="card"><strong>9</strong><span>observation masks</span></div>
    <div class="card"><strong>7</strong><span>PDE loss reports</span></div>
  </div>
  <div class="note"><b>Scientific status:</b> the code and quality-reporting workflow are available. Paper-grade ground truth still requires the documented numerical-validation gate.</div>
</section>

<section id="library" class="section library-teaser">
  <div class="smallcaps">Supporting resource</div>
  <h2>AI4PDE paper library</h2>
  <p class="muted">The benchmark comes first. The literature library is a supporting index of <b>{n_total}</b> AI4PDE/AI4SDE papers, with <b>{n_curated}</b> detailed pages. Find work by the physical equation or by the learning method.</p>
  <div class="library-links">
    <a class="card" href="pde-problems/"><b>Browse by PDE</b><span>Fluids, diffusion, elliptic systems, waves, and more</span></a>
    <a class="card" href="baselines/"><b>Browse by method</b><span>Neural operators, PINNs, diffusion, graphs, and baselines</span></a>
    <a class="card" href="research/"><b>Search all papers</b><span>Combine PDE, method, venue, year, and keyword filters</span></a>
  </div>
</section>
"""

    return page(
        title="PDE-OBS Benchmark Builder — PartialObs–PDEBench",
        root=root,
        current="home",
        hero_h1="PDE-OBS Benchmark Builder",
        hero_subtitle_html=(
            "Design reproducible <b>partial-observation PDE benchmarks</b>, generate "
            "the exact run code, and report data quality for every selected equation."
        ),
        hero_meta_html=(
            '<div class="meta">'
            '  <div><b>Project:</b> <a class="meta-link" href="https://ru1ch3n.github.io/PartialObs--PDEBench" target="_blank" rel="noopener noreferrer">ru1ch3n.github.io/PartialObs--PDEBench</a></div>'
            '  <div><b>Source:</b> private GitHub repository; this public site is a static release mirror.</div>'
            "</div>"
        ),
        hero_card_html=hero_card,
        body_html=body,
    )


def render_progress() -> str:
    progress = FULL_DATASET_PROGRESS
    generation = progress["generation"]
    artifacts = progress["artifacts"]
    quality = progress["quality"]
    final_qc = progress["final_qc"]
    plans = progress["plans"]

    body = f"""
<section class="section progress-hero-panel">
  <div class="status-line"><span class="status-dot running" aria-hidden="true"></span><b>{html_escape(progress['status_label'])}</b></div>
  <p class="muted">Snapshot updated {html_escape(progress['updated_at'])}. Generation and artifact integrity are complete; release acceptance remains pending until the separate strict aggregate job writes and validates its summary.</p>
  <div class="progress-meter" aria-label="Generation completion"><span style="width: {generation['percentage']:.0f}%"></span></div>
  <div class="progress-kpis">
    <div class="card"><strong>{generation['shards_complete']:,}</strong><span>of {generation['shards_expected']:,} shards</span></div>
    <div class="card"><strong>{generation['samples_complete']:,}</strong><span>of {generation['samples_expected']:,} samples</span></div>
    <div class="card"><strong>{generation['pde_families']}</strong><span>PDE families</span></div>
    <div class="card"><strong>{generation['percentage']:.0f}%</strong><span>generation complete</span></div>
  </div>
</section>

<section class="section">
  <div class="smallcaps">Artifact integrity</div>
  <h2>Every expected shard and sidecar is present</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Check</th><th>Verified result</th></tr></thead>
    <tbody>
      <tr><td>HDF5 / manifest / SHA256 / metadata / quality</td><td>{artifacts['hdf5_complete']:,} complete sets</td></tr>
      <tr><td>Missing or mismatched expected rows</td><td>{artifacts['missing_or_mismatched']}</td></tr>
      <tr><td>Partials / live locks</td><td>{artifacts['partials']} / {artifacts['locks']}</td></tr>
      <tr><td>Live generation / duplicate outputs</td><td>{artifacts['live_generation_jobs']} / {artifacts['duplicate_or_overlapping_outputs']}</td></tr>
      <tr><td>Approximate storage</td><td>{generation['storage_gib_approx']} GiB</td></tr>
    </tbody>
  </table></div>
</section>

<section class="section">
  <div class="smallcaps">Strict scientific gate</div>
  <h2>All shard-level PDE checks pass without relaxing the threshold</h2>
  <div class="quality-callout">
    <div><span>Maximum accepted <code>pde_loss</code></span><strong>{quality['max_pde_loss']:.17f}</strong></div>
    <div><span>Required ceiling</span><strong>{quality['gate']:.2f}</strong></div>
    <div><span>R4 recovery maximum</span><strong>{quality['r4_max_pde_loss']:.17f}</strong></div>
  </div>
  <p>No model training was run. Recovery preserved logical sample identities and stored T=15, used exact-seed diagnostics, and produced non-overlapping replacements only for absent or rejected rows.</p>
</section>

<section class="section">
  <div class="smallcaps">Final aggregate validation</div>
  <h2>One strict QC job is reading all 3,360 shards</h2>
  <div class="timeline">
    <div class="done"><b>Generation complete</b><span>Jobs 2130068 and 2130070 completed all six R4 rows with verified sidecars and checksums.</span></div>
    <div class="done"><b>Read-only full audit complete</b><span>560,000 samples, seven PDEs &times; 80,000, zero missing rows, partials, locks, or overlaps.</span></div>
    <div class="done"><b>Expected-plan reconciliation complete</b><span>Job 2130270 found exactly two stale Burgers <code>time_steps</code> values in the expected plan, not a data-quality failure. The original plan remains unchanged and the reconciled QC plan has zero content-spec mismatches.</span></div>
    <div class="running"><b>Strict QC job {final_qc['job_id']} running</b><span>{html_escape(final_qc['partition'])}; {html_escape(final_qc['resources'])}; validates shards, requires all PDEs, and enforces <code>pde_loss &le; {final_qc['max_pde_loss']:.2f}</code>.</span></div>
    <div><b>Release acceptance pending</b><span>Accepted only after <code>summary.json</code> proves 3,360 shards / 560,000 samples and strict quality success.</span></div>
  </div>
</section>

<section class="section">
  <div class="smallcaps">Reproducibility anchors</div>
  <h2>Frozen and reconciled plan identities</h2>
  <div class="cards">
    <div class="card"><h3>Original combined plan</h3><p><code>{html_escape(plans['original_combined']['name'])}</code></p><p class="hash">SHA256 {html_escape(plans['original_combined']['sha256'])}</p><p class="muted">Preserved unchanged.</p></div>
    <div class="card"><h3>Final-QC expected plan</h3><p><code>{html_escape(plans['final_qc']['name'])}</code></p><p class="hash">SHA256 {html_escape(plans['final_qc']['sha256'])}</p><p class="muted">Changes only two documented R2 Burgers internal time-grid values; 3,360 unique rows / 560,000 samples.</p></div>
  </div>
  <p><a href="../assets/progress.json">Download the machine-readable progress snapshot</a>.</p>
</section>
"""

    return page(
        title="Full Dataset Progress — PDE-OBS",
        root="../",
        current="progress",
        hero_h1="PDE-OBS Full Dataset Progress",
        hero_subtitle_html=(
            "A public, auditable view of generation completeness, artifact integrity, "
            "strict quality, recovery provenance, and final aggregate validation."
        ),
        hero_meta_html=(
            '<div class="meta"><div><b>Campaign:</b> numerics-full-t15-6c7c7e31</div>'
            '<div><b>Stored trajectory:</b> T=15</div></div>'
        ),
        body_html=body,
    )


def render_research_index(papers: List[Dict[str, Any]]) -> str:
    """Research hub page.

    IMPORTANT: this page is rendered client-side from docs/assets/papers_db.json
    to keep the HTML small and scalable (thousands of papers).
    """
    root = "../"  # from docs/research/index.html
    n_total = len(papers)
    n_curated = sum(1 for p in papers if str(p.get("status") or "index") == "curated")

    body = f"""
<section class="section">
  <div class="smallcaps">Supporting resource</div>
  <h2>Find AI4PDE papers</h2>
  <p class="muted">
    Search <b>{n_total}</b> papers, starting with the physical PDE or the method family.
    Curated pages add experiment and baseline notes; index pages are bibliographic placeholders.
  </p>

  <div class="library-index-links">
    <a class="card" href="../pde-problems/"><b>Browse the PDE index</b><span>Group papers by equation family</span></a>
    <a class="card" href="../baselines/"><b>Browse the method index</b><span>Group papers by learning approach</span></a>
  </div>

  <div class="card" style="margin-top:16px;">
    <div class="grid">
      <div>
        <div class="smallcaps">Search</div>
        <input id="q" class="input" placeholder="Search title, author, PDE, or method..." />
      </div>

      <div>
        <div class="smallcaps">Filters</div>
        <div class="row">
          <select id="f_pde" class="select"><option value="">All PDEs</option></select>
          <select id="f_method" class="select"><option value="">All methods</option></select>
          <select id="f_venue" class="select"><option value="">All venues</option></select>
          <select id="f_status" class="select">
            <option value="">All statuses</option>
            <option value="curated">curated</option>
            <option value="index">index</option>
          </select>
        </div>
        <div class="muted" style="margin-top:8px;">
          Showing <span id="shownCount">0</span> / <span id="totalCount">{n_total}</span>
          (<b>{n_curated}</b> curated)
        </div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <div class="row" style="align-items:center; justify-content:space-between;">
      <div class="muted">
        Selected: <b><span id="selCount">0</span></b>
      </div>
      <div class="row">
        <button id="btnCopyBib" class="btn">Copy BibTeX</button>
        <button id="btnDownloadBib" class="btn primary">Download .bib</button>
        <button id="btnClearSel" class="btn">Clear</button>
      </div>
    </div>
  </div>

  <div class="tablewrap" style="margin-top:16px;">
    <table class="papers">
      <thead>
        <tr>
          <th style="width:60px;">Pick</th>
          <th style="width:80px;">Year</th>
          <th>Paper</th>
          <th style="width:140px;">Venue</th>
          <th style="width:170px;">Method</th>
          <th style="width:220px;">PDEs</th>
          <th style="width:220px;">Tasks</th>
          <th style="width:90px;">Status</th>
        </tr>
      </thead>
      <tbody id="paperRows"></tbody>
    </table>
  </div>

  <div class="note" style="margin-top:16px;">
    Tip: if you find an index placeholder you care about, click into it and use the <b>Contribute</b> tab to add a curated JSON summary.
  </div>
</section>
"""

    return page(
        title="AI4PDE Paper Library — PartialObs–PDEBench",
        root=root,
        current="research",
        hero_h1="AI4PDE Paper Library",
        hero_subtitle_html="Find papers by PDE, method, venue, or keyword. The library supports the benchmark and lives after the Builder workflow.",
        hero_meta_html="",
        hero_card_html="",
        extra_head=(
            "<script>window.PAPERS_DB_URL='../assets/papers_db.json';</script>"
            '<script defer src="../assets/research.js"></script>'
        ),
        body_html=body,
    )


def render_paper_placeholder() -> str:
    """Generic paper page for index placeholders.

    This avoids generating thousands of per-paper HTML files. The page loads
    docs/assets/papers_db.json and renders the requested paper by `?slug=...`.
    """
    root = "../../"  # docs/research/paper/index.html

    body = """
<section class="section">
  <div id="paperMount"></div>
</section>
"""

    return page(
        title="Paper",
        root=root,
        current="research",
        hero_h1="Paper",
        hero_subtitle_html="<span class='muted' id='paperSubtitle'>Loading…</span>",
        hero_meta_html="",
        hero_card_html="",
        extra_head=(
            "<script>window.PAPERS_DB_URL='../../assets/papers_db.json';</script>"
            '<script defer src="../../assets/paper.js"></script>'
        ),
        body_html=body,
    )


def render_pde_problems(papers: List[Dict[str, Any]]) -> str:
    root = "../"  # docs/pde-problems/index.html
    pde_to_papers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in papers:
        pdes_display, _ = get_display_list(p, "pdes")
        for pde in pdes_display:
            pde_to_papers[pde].append(p)

    def family(pde: str) -> str:
        t = pde.lower()
        if any(k in t for k in ["navier", "stokes", "turbulence", "kolmogorov", "fluid"]):
            return "Fluid dynamics"
        if any(k in t for k in ["darcy", "poisson", "helmholtz", "laplace", "elliptic"]):
            return "Elliptic PDEs"
        if any(k in t for k in ["wave", "hyperbolic"]):
            return "Hyperbolic PDEs"
        if any(k in t for k in ["heat", "diffusion", "parabolic"]):
            return "Parabolic PDEs"
        if any(k in t for k in ["reaction", "allen", "phase", "cahn"]):
            return "Reaction–diffusion / phase field"
        if "shallow water" in t or "geophysical" in t or "atmos" in t:
            return "Geophysical flows"
        if "maxwell" in t or "electrom" in t:
            return "Electromagnetics"
        return "Other"

    fam_to_pdes: Dict[str, List[str]] = defaultdict(list)
    for pde in pde_to_papers.keys():
        fam_to_pdes[family(pde)].append(pde)
    for k in fam_to_pdes:
        fam_to_pdes[k] = sorted(fam_to_pdes[k], key=lambda s: s.lower())

    sections: List[str] = []
    for fam in sorted(fam_to_pdes.keys()):
        rows: List[str] = []
        for pde in fam_to_pdes[fam]:
            ps = pde_to_papers[pde]
            method_counter = Counter([pp.get("method_class", "SciML") for pp in ps])
            top_methods = ", ".join([k for k, _ in method_counter.most_common(3)])

            base_counter = Counter()
            for pp in ps:
                for b in pp.get("baselines", []) or []:
                    name = b.split("(")[0].strip()
                    if name:
                        base_counter[name] += 1
            top_bases = ", ".join([k for k, _ in base_counter.most_common(5)])
            link = f"../research/?pde={quote(pde)}"
            rows.append(
                "<tr>"
                f"<td><b>{html_escape(pde)}</b></td>"
                f"<td>{len(ps)}</td>"
                f'<td class="muted">{html_escape(top_methods) if top_methods else "—"}</td>'
                f'<td class="muted">{html_escape(top_bases) if top_bases else "—"}</td>'
                f'<td><a href="{link}">View papers</a></td>'
                "</tr>"
            )

        sections.append(
            '<section class="section">'
            f"  <h2>{html_escape(fam)}</h2>"
            '  <div class="tablewrap"><table><thead><tr>'
            "    <th>PDE</th><th># papers</th><th>Common method classes</th><th>Common baselines (curated pages)</th><th></th>"
            "  </tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div>"
            "</section>"
        )

    body = (
        '<section class="section">'
        "  <h2>Browse by PDE problem</h2>"
        "  <p>This page groups PDEs into common families. Each row links to the Research table with a PDE filter applied.</p>"
        '  <div class="note">PDE tags for <b>index-only</b> papers are auto-extracted from titles and may be incomplete. Curated pages include better coverage.</div>'
        "</section>" + "\n".join(sections)
    )

    hero_card = (
        '<div class="hero-card">'
        '  <div class="smallcaps">How to use</div>'
        '  <p class="muted" style="margin-top:8px;">'
        "    Pick a PDE family → click <b>View papers</b> to jump into the Research table."
        "  </p>"
        "</div>"
    )

    return page(
        title="Browse AI4PDE Papers by PDE — PartialObs–PDEBench",
        root=root,
        current="pde",
        hero_h1="Browse papers by PDE",
        hero_subtitle_html="Start from the physical equation, then see the methods and baselines used in the literature.",
        hero_card_html=hero_card,
        body_html=body,
    )


def render_baselines(papers: List[Dict[str, Any]]) -> str:
    root = "../"  # docs/baselines/index.html

    # Method classes (across all papers)
    cls_counter = Counter([p.get("method_class", "SciML") for p in papers])
    cls_rows: List[str] = []
    for cls, n in sorted(cls_counter.items(), key=lambda x: (-x[1], x[0])):
        eq = METHOD_MATH.get(cls, METHOD_MATH.get("SciML", []))
        eq_html = render_math_block(eq[:2]) if eq else "—"
        link = f"../research/?method={quote(cls)}"
        cls_rows.append(
            "<tr>"
            f"<td><b>{html_escape(cls)}</b></td>"
            f"<td>{n}</td>"
            f"<td>{eq_html}</td>"
            f'<td><a href="{link}">View papers</a></td>'
            "</tr>"
        )
    cls_table = (
        '<div class="tablewrap"><table><thead><tr>'
        "<th>Method class</th><th># papers</th><th>Core objective (template)</th><th></th>"
        "</tr></thead><tbody>" + "\n".join(cls_rows) + "</tbody></table></div>"
    )

    # Baseline methods (from curated pages only)
    base_to_papers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in papers:
        for b in p.get("baselines", []) or []:
            name = b.split("(")[0].strip()
            if name:
                base_to_papers[name].append(p)

    base_rows: List[str] = []
    for base, ps in sorted(base_to_papers.items(), key=lambda kv: (-len(kv[1]), kv[0].lower())):
        ex = sorted(ps, key=lambda x: (-x.get("year", 0), x.get("short_title", "")))[:3]
        ex_links = ", ".join(
            f'<a href="{paper_link(pp, "../")}">{html_escape(pp["short_title"])}</a>' for pp in ex
        )
        qlink = f"../research/?q={quote(base)}"
        base_rows.append(
            "<tr>"
            f"<td><b>{html_escape(base)}</b></td>"
            f"<td>{len(ps)}</td>"
            f'<td class="muted">{ex_links or "—"}</td>'
            f'<td><a href="{qlink}">Search</a></td>'
            "</tr>"
        )
    base_table = (
        '<div class="tablewrap"><table><thead><tr>'
        "<th>Baseline method</th><th># curated papers</th><th>Examples</th><th></th>"
        "</tr></thead><tbody>" + "\n".join(base_rows) + "</tbody></table></div>"
    )

    body = (
        '<section class="section">'
        "  <h2>Method classes</h2>"
        "  <p>High-level taxonomy used across this website. Each row links to the Research table with a class filter.</p>"
        "</section>" + cls_table + '<section class="section">'
        "  <h2>Baseline methods (curated pages)</h2>"
        "  <p>Baseline lists are only available on manually curated paper pages. This table summarizes what is currently extracted.</p>"
        "</section>" + base_table
    )

    hero_card = (
        '<div class="hero-card">'
        '  <div class="smallcaps">Tip</div>'
        '  <p class="muted" style="margin-top:8px;">'
        "    Use <b>Method classes</b> to compare approaches, and <b>Baseline methods</b> to reproduce literature tables."
        "  </p>"
        "</div>"
    )

    return page(
        title="Browse AI4PDE Papers by Method — PartialObs–PDEBench",
        root=root,
        current="baselines",
        hero_h1="Browse papers by method",
        hero_subtitle_html="Start from a learning approach, then find the PDEs and comparison baselines used across papers.",
        hero_card_html=hero_card,
        extra_head=(
            "<script>window.MathJax={tex:{inlineMath:[['\\(','\\)'],['$','$']]}};</script>"
            '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
        ),
        body_html=body,
    )


def render_contribute(papers: List[Dict[str, Any]]) -> str:
    n_total = len(papers)
    n_curated = sum(1 for p in papers if p.get("status") == "curated")

    example_json = """{
  "slug": "my-paper-2025",
  "status": "curated",
  "full_title": "Full paper title goes here",
  "short_title": "MyPaper",
  "authors": "First Author; Second Author; ...",
  "year": 2025,
  "venue": "ICLR",
  "method_class": "Operator learning",
  "links": {
    "paper": "https://arxiv.org/abs/xxxx.xxxxx",
    "code": "https://github.com/user/repo"
  },

  "tldr": "2–4 sentences: what the method does + why it matters.",
  "problem": "What problem does the paper solve? (be concrete about partial observation / inverse / operator learning / etc.)",

  "contrib": [
    "Main contribution #1 (method idea).",
    "Main contribution #2 (training / inference trick).",
    "Main contribution #3 (benchmark / dataset / analysis)."
  ],
  "benefits": [
    "Why this is better than prior work (accuracy / speed / generalization / stability / data efficiency)."
  ],

  "core_math": [
    "Put the key equations here in LaTeX (no $...$ wrappers).",
    "Example: G_\\theta(u)(y) = \\sum_{k=1}^p b_k(u)\\,t_k(y)"
  ],

  "data_setting": [
    "Dataset: size, how generated, train/val/test split.",
    "PDE + domain + discretization / resolution.",
    "Observation pattern (mask/sensors) + noise model."
  ],
  "model_setting": [
    "Architecture (layers, width, latent dims, Fourier modes, etc.).",
    "Inputs/outputs parameterization (what is u, what is a, what is y)."
  ],
  "training_setting": [
    "Optimizer, learning rate schedule, epochs/steps, batch size, hardware."
  ],

  "baselines": [
    "Baseline A",
    "Baseline B"
  ],

  "results_tables": [
    {
      "title": "Main quantitative results (copy numbers from the paper tables)",
      "note": "Write the metric + what lower/higher means.",
      "header": ["Setting", "Method", "Metric"],
      "rows": [
        ["...", "MyPaper", "0.012"],
        ["...", "Baseline A", "0.034"]
      ]
    }
  ],

  "interesting": [
    "Any extra insights that are useful for readers (failure modes, ablations, theory notes, etc.)."
  ],

  "bibtex": "@inproceedings{...}"
}"""

    body = f"""
    <h1>Contribute</h1>
    <p class=\"muted\">Current DB: <b>{n_total}</b> papers (<b>{n_curated}</b> curated).</p>

    <div class=\"card\">
      <h2>How the data is stored</h2>
      <ul>
        <li><b>Index list (metadata)</b>: <code>scripts/research_db.ndjson</code> (JSON Lines). Add new papers here (title/authors/year/links) when you want them searchable.</li>
        <li><b>Curated summaries (rich content)</b>: <code>data/curations/&lt;slug&gt;.json</code>. Only add these for papers you want to curate deeply (tables, math, detailed settings).</li>
      </ul>
      <p class=\"muted\">Tip: keep most papers as <b>index-only</b>; curate a small set with high quality and lots of details.</p>
    </div>

    <div class=\"card\">
      <h2>Add a new curated paper (step-by-step)</h2>
      <ol>
        <li>Find the paper in <a href=\"../research/\">Research</a>. Open its page (index view) and copy the <b>slug</b> from the URL (<code>?slug=...</code>).</li>
        <li>Create <code>data/curations/&lt;slug&gt;.json</code> using the template below.</li>
        <li>Fill in the fields. For <b>results_tables</b>, please copy the numbers from the paper’s tables (metrics, settings, baselines). For <b>core_math</b>, include the core idea + equations in LaTeX.</li>
        <li>Rebuild the website: <code>python scripts/generate_research_site.py</code></li>
        <li>Commit and push. GitHub Pages will serve <code>docs/</code>.</li>
      </ol>
    </div>

    <div class=\"card\">
      <h2>Template (copy/paste)</h2>
      <pre><code>{html_escape(example_json)}</code></pre>
      <p class=\"muted\">You can add extra fields if useful; unknown fields are ignored by the site generator.</p>
    </div>

    <div class=\"card\">
      <h2>Batch BibTeX export</h2>
      <p>On the <a href=\"../research/\">Research</a> page you can use the <b>Pick</b> checkboxes to select many papers and export a BibTeX file (copy or download).</p>
    </div>

    <div class=\"card\">
      <h2>Bulk import (optional)</h2>
      <p>If you have a BibTeX file and want to convert it into index entries (NDJSON), use:</p>
      <pre><code>python scripts/import_bibtex_to_json.py path/to/papers.bib</code></pre>
      <p class=\"muted\">This script is best-effort and produces metadata. Curations still require human-written JSON files.</p>
    </div>
    """

    return page(
        title="Contribute",
        root="../",
        current="contribute",
        hero_h1="Contribute",
        hero_subtitle_html=f"Current DB: <b>{n_total}</b> papers (<b>{n_curated}</b> curated).",
        hero_meta_html="",
        hero_card_html="",
        body_html=body,
    )


def benchmark_builder_options() -> Dict[str, Any]:
    """Build browser-safe choices directly from the frozen benchmark contract."""

    source_root = str(REPO_ROOT / "src")
    added_path = source_root not in sys.path
    if added_path:
        sys.path.insert(0, source_root)
    try:
        from pdeobs.methods import METHOD_REGISTRY, available_methods
        from pdeobs.protocol import benchmark_contract
        from pdeobs.splits import tier_regime_counts

        contract = benchmark_contract()
        method_labels = {
            "fno": "FNO",
            "unet": "U-Net",
            "cno": "CNO-like",
            "mae_small": "MAE small",
            "rbf": "RBF interpolation",
        }
        registered_methods = set(available_methods())
        method_rows = []
        for method_name in sorted(registered_methods):
            factory = METHOD_REGISTRY[method_name]
            capabilities = getattr(factory, "capabilities", None)
            method_rows.append(
                {
                    "value": method_name,
                    "label": method_labels.get(method_name, method_name.replace("_", " ").title()),
                    "tasks": sorted(getattr(capabilities, "tasks", ())),
                    "capabilities_known": capabilities is not None,
                    "trainable": bool(getattr(capabilities, "trainable", False)),
                    "temporal": bool(getattr(capabilities, "temporal", False)),
                    "supports_multichannel": bool(
                        getattr(capabilities, "supports_multichannel", False)
                    ),
                    "reference_only": bool(getattr(capabilities, "reference_only", True)),
                    "note": str(
                        getattr(
                            capabilities,
                            "notes",
                            "No machine-readable capabilities are registered; use an explicit config.",
                        )
                    ),
                }
            )

        observation_training = contract["observation_training"]
        protocol_methods = []
        for row in observation_training["methods"]:
            registry_name = row.get("registry_name")
            declared_executable = str(row["execution_status"]).startswith("executable_")
            builder_available = bool(
                declared_executable and registry_name and registry_name in registered_methods
            )
            protocol_methods.append(
                {
                    **row,
                    "value": row["method_id"],
                    "default_seeds": 3 if row["method_id"] == "pinn_or_pino" else 1,
                    "variant_required": bool(row.get("implementation_choice_required", False)),
                    "builder_available": builder_available,
                    "command_generation": ("single_run_only" if builder_available else "blocked"),
                }
            )
    finally:
        if added_path:
            sys.path.remove(source_root)

    pde_details = {
        "darcy": (
            "Darcy flow",
            "-div(a grad u) - f",
            "Variable-coefficient flux residual with stored or identified forcing.",
        ),
        "poisson": ("Poisson", "-Laplacian(u) - f", "Elliptic equation residual."),
        "helmholtz": (
            "Helmholtz",
            "(-Laplacian - k^2)u - f",
            "Nominal PDE residual; regularized transfer defect is reported separately.",
        ),
        "heat": ("Heat", "u_t - D Laplacian(u)", "Saved-frame temporal residual."),
        "reaction_diffusion": (
            "Reaction-diffusion",
            "u_t - D Laplacian(u) - r(u - u^3)",
            "Allen-Cahn-type temporal residual.",
        ),
        "burgers": (
            "2-D Burgers",
            "u_t + u(u_x + u_y) - nu Laplacian(u)",
            "Temporal transport-diffusion residual.",
        ),
        "navier_stokes": (
            "Navier-Stokes",
            "omega_t + velocity dot grad(omega) - nu Laplacian(omega)",
            "Periodic, rectangular bounded, and obstacle routes store vorticity and use their registered spectral, DST, or masked-streamfunction reconstruction; divergence and boundary losses are reported separately.",
        ),
    }
    boundaries = {
        "dirichlet": ("Dirichlet / no-slip", "Prescribed boundary value."),
        "neumann": ("Neumann / free-slip", "Prescribed normal derivative."),
        "periodic": ("Periodic", "Opposite-edge continuity."),
        "robin_obstacle": (
            "Robin / obstacle",
            "Family-conditioned mixed boundary or embedded obstacle.",
        ),
    }
    task_labels = {
        "sparse_recovery": "Sparse recovery",
        "forward_prediction": "Forward prediction",
        "inverse_prediction": "Inverse prediction",
        "semantic_retrieval": "Semantic retrieval",
        "world_modeling": "World modeling / rollout",
        "solver_routing": "Solver routing",
        "foundation_transfer": "Foundation transfer",
    }
    mask_labels = {
        "random_1pct": "Random 1%",
        "random_3pct": "Random 3% (500 points at 128x128 training)",
        "random_5pct": "Random 5%",
        "random_10pct": "Random 10%",
        "regular_grid": "Regular grid",
        "block_missing": "Missing block",
        "line_sensors": "Line sensors",
        "boundary_sensors": "Boundary sensors",
        "clustered_sensors": "Clustered sensors",
    }
    dataset = contract["dataset"]
    tiers = []
    for name, row in dataset["tiers"].items():
        size = int(row["samples_per_macro_case"])
        tiers.append(
            {
                "value": name,
                "label": f"{name.title()} ({size:,} samples / macro case)",
                "samples_per_macro_case": size,
                "full_matrix_samples": int(row["total_samples"]),
                "regime_counts": tier_regime_counts(size),
            }
        )

    all_protocol_methods = [row["value"] for row in protocol_methods]
    all_observations = list(contract["masks"])
    three_anchor_observations = ["random_1pct", "random_3pct", "block_missing"]
    protocol_by_id = {row["value"]: row for row in protocol_methods}
    observation_accounting = observation_training["campaign_accounting"]
    compute_planning = observation_training["compute_planning"]

    def campaign_counts(method_observations: dict[str, list[str]]) -> dict[str, int]:
        pde_count = int(observation_training["dataset_accounting"]["pde_count"])
        result_cells = 0
        neural_jobs = 0
        pod_fits = 0
        prior_jobs = 0
        raw_evaluations = 0
        for method_id, observations in method_observations.items():
            row = protocol_by_id[method_id]
            seeds = int(row["default_seeds"])
            observation_count = len(observations)
            cells = pde_count * observation_count
            result_cells += cells
            raw_evaluations += cells * seeds
            fit_scope = row["fit_scope"]
            if fit_scope == "once_per_pde_and_observation":
                neural_jobs += cells * seeds
            elif fit_scope == "once_per_pde_training_split":
                pod_fits += pde_count
            elif fit_scope == "once_per_pde_prior":
                prior_jobs += pde_count
        preparation_min = neural_jobs + pod_fits
        return {
            "result_cells": result_cells,
            "neural_training_jobs": neural_jobs,
            "pod_fit_jobs": pod_fits,
            "prior_preparation_jobs_min": 0,
            "prior_preparation_jobs_max": prior_jobs,
            "preparation_jobs_min": preparation_min,
            "preparation_jobs_max": preparation_min + prior_jobs,
            "raw_evaluation_runs": raw_evaluations,
        }

    full_anchor_methods = [
        "rbf",
        "gappy_pod",
        "unet",
        "fno",
        "cno",
        "diffusionpde",
        "fundps",
    ]
    full_hybrid_method_observations = {
        method: (all_observations if method in {"rbf", "gappy_pod"} else three_anchor_observations)
        for method in full_anchor_methods
    }
    all_method_observations = {method: all_observations for method in all_protocol_methods}
    campaign_presets = [
        {
            "value": "medium_recommended",
            "label": "Medium 140k: full 10-method comparison (recommended)",
            "tier": "medium",
            "methods": all_protocol_methods,
            "method_observations": all_method_observations,
            **campaign_counts(all_method_observations),
            "gpu_hours_low": compute_planning["medium_gpu_hours_pinn_or_pino_three_seeds"][0],
            "gpu_hours_high": compute_planning["medium_gpu_hours_pinn_or_pino_three_seeds"][1],
            "recommendation": "Feasible but tight after a measured runtime pilot.",
        },
        {
            "value": "full_anchor_hybrid",
            "label": "Full 560k: hybrid anchor matrix (recommended full-tier check)",
            "tier": "full",
            "methods": full_anchor_methods,
            "method_observations": full_hybrid_method_observations,
            **campaign_counts(full_hybrid_method_observations),
            "gpu_hours_low": None,
            "gpu_hours_high": None,
            "recommendation": "Run a hardware-specific pilot before reserving this campaign.",
        },
        {
            "value": "full_all_methods_planning",
            "label": "Full 560k: all methods and observations (not recommended)",
            "tier": "full",
            "methods": all_protocol_methods,
            "method_observations": all_method_observations,
            **campaign_counts(all_method_observations),
            "gpu_hours_low": compute_planning["full_gpu_hours_pinn_or_pino_three_seeds"][0],
            "gpu_hours_high": compute_planning["full_gpu_hours_pinn_or_pino_three_seeds"][1],
            "recommendation": "Not feasible within the default ten-day planning budget.",
        },
    ]
    for preset in campaign_presets:
        preset["blocked_methods"] = [
            method_id
            for method_id in preset["methods"]
            if not protocol_by_id[method_id]["builder_available"]
        ]
        preset["execution_status"] = "planning_only_blocked"

    return {
        "schema_version": "pdeobs.benchmark-builder/v2",
        "generated_from": {
            "contract": contract["schema_version"],
            "default_config": "configs/dataset/default.yaml",
        },
        "release": contract["publication_gate"],
        "pdes": [
            {
                "value": value,
                "label": pde_details[value][0],
                "loss": pde_details[value][1],
                "note": pde_details[value][2],
            }
            for value in dataset["pde_families"]
        ],
        "boundaries": [
            {"value": value, "label": boundaries[value][0], "note": boundaries[value][1]}
            for value in dataset["boundaries"]
        ],
        "settings": [
            {"value": value, "label": value.replace("_", " ").title()}
            for value in dataset["settings"]
        ],
        "regimes": [{"value": value, "label": value.title()} for value in dataset["regimes"]],
        "tiers": tiers,
        "tasks": [
            {
                "value": task["name"],
                "label": task_labels[task["name"]],
                "status": task["status"],
                "metrics": task["metrics"],
            }
            for task in contract["tasks"]
        ],
        "splits": [
            {"value": value, "label": value.replace("_", " ").upper()}
            for value in contract["splits"]
        ],
        "masks": [{"value": value, "label": mask_labels[value]} for value in contract["masks"]],
        "models": method_rows,
        "protocol_methods": protocol_methods,
        "campaign_planner": {
            "schema_version": observation_training["schema_version"],
            "protocol_url": "https://github.com/ru1ch3n/PartialObs--PDEBench/blob/main/docs/OBSERVATION_TRAINING_PROTOCOL.md",
            "scope": observation_accounting["scope"],
            "pde_count": observation_training["dataset_accounting"]["pde_count"],
            "observation_count": observation_training["dataset_accounting"]["observation_count"],
            "primary": observation_training["primary"],
            "secondary": observation_training["secondary"],
            "dataset_accounting": observation_training["dataset_accounting"],
            "observation_counts_128": observation_training["observation_counts_128"],
            "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "budget": compute_planning,
            "scientific_caveats": observation_training["scientific_caveats"],
            "presets": campaign_presets,
        },
        "environments": [
            {"value": "local", "label": "Linux/macOS/local Bash"},
            {"value": "server", "label": "Linux server"},
            {"value": "seawulf", "label": "SeaWulf (Slurm)"},
        ],
        "quality_profiles": [
            {
                "value": "report",
                "label": "Report",
                "note": "Compute every available per-sample PDE/physics loss without applying an unfrozen scientific threshold; malformed array or geometry contracts are still quarantined.",
            },
            {
                "value": "strict",
                "label": "Strict generation gate",
                "note": "Reject non-finite, geometry, initial-condition, and boundary failures; an optional calibrated PDE-loss limit adds a residual gate.",
            },
            {
                "value": "publication",
                "label": "Publication-candidate quality gate",
                "note": "Expert-only and blocked by the Builder: requires all seven families, an external trusted verifier for solver evidence, and a per-stratum frozen threshold table. The current package intentionally keeps the candidate gate and publication_ready false.",
            },
        ],
        "quality_outputs": [
            "Per-sample quality records in HDF5 metadata",
            "Per-shard *.quality.json summaries",
            "Strict rejected-sample records in *.quality-failures.jsonl",
            "Aggregate summary.quality.json and summary.quality.csv reports",
            "Standalone pdeobs quality output at the explicitly selected report path",
            "Manifest/checksum validation and pass/warn/fail gate status",
        ],
    }


def render_builder() -> str:
    """Render the client-side benchmark dataset and quality command builder."""

    body = """
<section class="section builder-intro">
  <p class="builder-lead">Choose a benchmark slice, inspect its size, and copy the exact dataset-generation code. Ground truth is generated once; observation masks are deterministic views applied afterward.</p>
  <div class="note"><b>Scientific gate:</b> every run produces PDE-specific quality diagnostics. Bundled reference solvers are not paper-grade ground truth until the documented numerical-validation and release gates pass.</div>
</section>

<section class="builder-layout" aria-labelledby="builder-form-title">
  <form id="benchmark-builder" class="card builder-controls">
    <h2 id="builder-form-title">1. Choose the benchmark</h2>
    <div class="builder-fields">
      <label>Environment<select id="builder-environment" class="select" data-option="environments" data-default="local"></select></label>
      <label>Release tier<select id="builder-tier" class="select" data-option="tiers" data-default="signal"></select></label>
      <label>PDE family<select id="builder-pde" class="select" data-option="pdes" data-all-label="All 7 PDE families" data-default="all"></select></label>
      <label>Boundary<select id="builder-boundary" class="select" data-option="boundaries" data-all-label="All 4 boundaries" data-default="all"></select></label>
      <label>Condition setting<select id="builder-setting" class="select" data-option="settings" data-all-label="All 10 settings" data-default="all"></select></label>
      <label>Physical regime<select id="builder-regime" class="select" data-option="regimes" data-all-label="All 3 regimes" data-default="all"></select></label>
      <label>Benchmark task<select id="builder-task" class="select" data-option="tasks" data-default="sparse_recovery"></select></label>
      <label>Observation mask<select id="builder-mask" class="select" data-option="masks" data-default="random_3pct"></select></label>
      <label>Evaluation split<select id="builder-split" class="select" data-option="splits" data-default="iid"></select></label>
      <label>Anchor model<select id="builder-model" class="select" data-option="models" data-default="fno"></select></label>
      <label>Quality profile<select id="builder-quality" class="select" data-option="quality_profiles" data-default="report"></select></label>
      <label>Calibrated max normalized PDE loss
        <input id="builder-threshold" class="input" inputmode="decimal" type="number" min="0" step="any" placeholder="Optional; required for publication candidate" />
      </label>
    </div>
    <details class="builder-advanced">
      <summary>Advanced paths and resources</summary>
      <div class="builder-fields">
        <label>Dataset name<input id="builder-name" class="input" value="pdeobs-custom" maxlength="64" /></label>
        <label>Data root<input id="builder-data-root" class="input" value="datasets" maxlength="240" /></label>
        <label>Run root<input id="builder-run-root" class="input" value="runs" maxlength="240" /></label>
        <label>Local workers<input id="builder-workers" class="input" type="number" min="1" max="128" step="1" value="4" /></label>
        <label>Samples per shard<input id="builder-shard-size" class="input" type="number" min="1" max="2000" step="1" value="700" /></label>
        <label>SeaWulf group<input id="builder-group" class="input" value="YOUR_GROUP" maxlength="80" /></label>
      </div>
    </details>
    <div class="builder-actions">
      <button class="btn" type="button" id="builder-reset">Reset choices</button>
      <span id="builder-state" class="muted" role="status" aria-live="polite"></span>
    </div>
  </form>

  <aside class="card builder-summary" aria-labelledby="builder-summary-title">
    <h2 id="builder-summary-title">Plan preview</h2>
    <dl class="builder-stats">
      <div><dt>Macro cases</dt><dd id="builder-macro-cases">-</dd></div>
      <div><dt>Samples</dt><dd id="builder-samples">-</dd></div>
      <div><dt>Shard jobs</dt><dd id="builder-jobs">-</dd></div>
      <div><dt>Task status</dt><dd id="builder-task-status">-</dd></div>
    </dl>
    <p id="builder-profile-note" class="muted"></p>
    <div id="builder-warning-box" class="builder-warning" role="status" aria-live="polite">
      <b>Checks before running</b>
      <ul id="builder-warnings"></ul>
    </div>
  </aside>
</section>

<details id="observation-training" class="section optional-campaign">
  <summary>Optional: future model campaign planning</summary>
  <div class="optional-campaign-body">
  <h2 id="campaign-title">Future model campaign planning</h2>
  <p><b>Primary matched-mask comparison:</b> every normal learned baseline gets an independent checkpoint for each PDE and each of the nine observation masks. The model trained on random 3% is reused only for the separate mask-transfer/OOD table; it never replaces matched-mask training in the primary IID table.</p>
  <p><a href="https://github.com/ru1ch3n/PartialObs--PDEBench/blob/main/docs/OBSERVATION_TRAINING_PROTOCOL.md">Open the complete observation-training protocol</a></p>
  <div class="note"><b>Execution boundary:</b> this planner counts the complete paper matrix, including methods that are not integrated in PDE-OBS. Its campaign manifest is planning-only. It never invents commands for Gappy POD, DeepONet, PINN/PINO, Transolver/GNOT, DiffusionPDE, or FunDPS.</div>

  <div class="campaign-layout">
    <div class="card campaign-controls">
      <h3>Campaign preset</h3>
      <div class="builder-fields">
        <label>Comparison matrix<select id="campaign-preset" class="select" data-option="campaign_presets" data-default="medium_recommended"></select></label>
        <label>Dedicated GPUs<input id="campaign-gpus" class="input" type="number" min="1" max="512" step="1" value="12" /></label>
        <label>Campaign days<input id="campaign-days" class="input" type="number" min="1" max="365" step="1" value="10" /></label>
        <label>Low utilization (%)<input id="campaign-utilization-low" class="input" type="number" min="1" max="100" step="1" value="75" /></label>
        <label>High utilization (%)<input id="campaign-utilization-high" class="input" type="number" min="1" max="100" step="1" value="80" /></label>
      </div>
      <p id="campaign-recommendation" class="muted"></p>
      <div class="builder-warning" role="status" aria-live="polite">
        <b>Planning status</b>
        <ul id="campaign-warnings"></ul>
      </div>
    </div>

    <aside class="card campaign-summary" aria-labelledby="campaign-summary-title">
      <h3 id="campaign-summary-title">Count and budget preview</h3>
      <dl class="builder-stats campaign-stats">
        <div><dt>Data pool</dt><dd id="campaign-data-pool">-</dd></div>
        <div><dt>Pool / PDE</dt><dd id="campaign-data-per-pde">-</dd></div>
        <div><dt>Train / PDE</dt><dd id="campaign-train-per-pde">-</dd></div>
        <div><dt>Result cells</dt><dd id="campaign-result-cells">-</dd></div>
        <div><dt>Preparation jobs</dt><dd id="campaign-preparation-jobs">-</dd></div>
        <div><dt>Raw evaluations</dt><dd id="campaign-evaluation-runs">-</dd></div>
        <div><dt>A6000 estimate</dt><dd id="campaign-gpu-estimate">-</dd></div>
        <div><dt>Safe capacity</dt><dd id="campaign-safe-capacity">-</dd></div>
      </dl>
      <p id="campaign-feasibility" class="campaign-feasibility" role="status" aria-live="polite"></p>
    </aside>
  </div>

  <div class="tablewrap campaign-table-wrap">
    <table class="campaign-method-table">
      <thead><tr><th>Method row</th><th>Fit policy</th><th>Seeds</th><th>Observations</th><th>Preparation</th><th>Evaluations</th><th>Builder status</th></tr></thead>
      <tbody id="campaign-method-body"></tbody>
    </table>
  </div>
</div>
</details>

<section class="section" aria-labelledby="builder-code-title">
  <div class="builder-code-heading">
    <div><h2 id="builder-code-title">2. Copy the generated code</h2><p class="muted">The dataset YAML is the executable source of truth. Use the environment tab that matches your machine.</p></div>
    <div class="builder-actions">
      <button id="builder-copy" class="btn primary" type="button">Copy current tab</button>
      <button id="builder-download" class="btn" type="button">Download YAML</button>
    </div>
  </div>
  <div class="builder-tabs" role="tablist" aria-label="Generated code">
    <button class="btn" id="builder-tab-setup" type="button" role="tab" aria-selected="true" aria-controls="builder-code-panel" data-builder-tab="setup">Setup</button>
    <button class="btn" id="builder-tab-yaml" type="button" role="tab" aria-selected="false" aria-controls="builder-code-panel" data-builder-tab="yaml">Dataset YAML</button>
    <button class="btn" id="builder-tab-run" type="button" role="tab" aria-selected="false" aria-controls="builder-code-panel" data-builder-tab="run">Generate + quality</button>
    <button class="btn" id="builder-tab-seawulf" type="button" role="tab" aria-selected="false" aria-controls="builder-code-panel" data-builder-tab="seawulf">SeaWulf chain</button>
    <button class="btn" id="builder-tab-campaign" type="button" role="tab" aria-selected="false" aria-controls="builder-code-panel" data-builder-tab="campaign">Campaign plan (not executable)</button>
  </div>
  <div id="builder-code-panel" role="tabpanel" tabindex="0" aria-labelledby="builder-tab-setup">
    <pre class="builder-code"><code id="builder-code">Loading benchmark contract...</code></pre>
  </div>
  <p id="builder-copy-status" class="muted" role="status" aria-live="polite"></p>
  <noscript><div class="note">JavaScript is required only for generating tailored commands. The static benchmark and server guides remain available without it.</div></noscript>
</section>

<section class="section" aria-labelledby="quality-coverage-title">
  <h2 id="quality-coverage-title">3. Quality report for every PDE</h2>
  <p>Every generated sample receives finite-value, geometry, initial/boundary, and family-specific physics diagnostics. Dataset aggregation reports each selected PDE separately; the complete benchmark reports all seven losses together.</p>
  <div class="tablewrap">
    <table class="builder-quality-table">
      <thead><tr><th>PDE</th><th>Reported normalized loss</th><th>Interpretation</th></tr></thead>
      <tbody id="builder-quality-body"></tbody>
    </table>
  </div>
  <div class="grid2">
    <div class="card"><h3>Always produced</h3><ul id="builder-quality-outputs"></ul></div>
    <div class="card"><h3>Gate levels</h3><ol><li><b>Report:</b> report scientific losses without applying an unfrozen PDE threshold; malformed array/geometry contracts are still quarantined.</li><li><b>Strict:</b> reject structural/BC/IC failures, log them to <code>*.quality-failures.jsonl</code>, and apply a PDE limit only when calibrated.</li><li><b>Publication candidate:</b> expert-only and intentionally blocked by this Builder until validated solver evidence and a per-stratum threshold table exist. It does not by itself set <code>publication_ready</code>.</li></ol></div>
  </div>
  <div class="note"><b>Important interpretation:</b> saved-frame residuals are not the same as integrator replay error. Helmholtz nominal residual and legacy regularized transfer defect stay separate. Bounded Navier-Stokes uses a versioned vorticity/streamfunction residual plus divergence and boundary diagnostics.</div>
</section>
"""
    return page(
        title="Benchmark Builder - PDE-OBS",
        root="../",
        current="builder",
        hero_h1="Build a partial-observation benchmark",
        hero_subtitle_html=(
            "Choose the physical system and observation view, then generate reproducible "
            "dataset and quality-control code."
        ),
        hero_meta_html=badges(
            ["7 PDE losses", "Local + Linux", "SeaWulf Slurm", "Deep-linkable choices"]
        ),
        extra_head=(
            '<script defer src="../assets/benchmark-builder.js?'
            'v=2026-08-16-observation-v3"></script>'
        ),
        body_html=body,
    )


def render_server() -> str:
    """Render the public Linux-server and SeaWulf quick-start page."""

    root = "../"
    linux_guide = "https://github.com/ru1ch3n/PartialObs--PDEBench/blob/main/docs/SERVER.md"
    seawulf_guide = (
        "https://github.com/ru1ch3n/PartialObs--PDEBench/blob/main/hpc/seawulf/README.md"
    )
    observation_training_guide = (
        "https://github.com/ru1ch3n/PartialObs--PDEBench/blob/main/"
        "docs/OBSERVATION_TRAINING_PROTOCOL.md"
    )
    body = f"""
<section class="section">
  <h2>Choose the machine</h2>
  <p>Need a custom factor slice? Use the <a href="../builder/">Benchmark Builder</a> first to generate matching YAML, quality gates, and local or SeaWulf commands.</p>
  <p>Planning the paper comparison? Read the <a href="{observation_training_guide}">matched-mask observation-training protocol</a> before submitting GPU jobs; the random 3% checkpoint belongs to a separate transfer/OOD table.</p>
  <div class="grid2">
    <div class="card">
      <h3>Single Linux server</h3>
      <p>Use a virtual environment and <code>tmux</code>. Keep datasets and runs outside Git, then begin with the two-sample smoke workflow.</p>
      <p><a href="{linux_guide}">Open the complete Linux server guide</a></p>
    </div>
    <div class="card">
      <h3>SeaWulf cluster</h3>
      <p>Pin one Git commit, build inside an allocation, and chain generation, validation, training, and evaluation with Slurm dependencies.</p>
      <p><a href="{seawulf_guide}">Open the complete SeaWulf guide</a></p>
    </div>
  </div>
</section>

<section id="linux" class="section">
  <h2>Linux server: verified smoke run</h2>
  <p>Run these commands after connecting over SSH. Keep the session alive with <code>tmux</code> before starting longer work.</p>
  <div class="card">
<pre><code>git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git
cd PartialObs--PDEBench
git checkout YOUR_RELEASE_TAG_OR_COMMIT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[train,test]"

export PDEOBS_DATA="$PWD/datasets"
export PDEOBS_RUNS="$PWD/runs"
mkdir -p "$PDEOBS_DATA" "$PDEOBS_RUNS"
pdeobs doctor
pdeobs protocol --check

tmux new -s pdeobs
pdeobs generate --config configs/dataset/smoke.yaml \\
  --output "$PDEOBS_DATA/smoke"
pdeobs aggregate --input "$PDEOBS_DATA/smoke" \\
  --output "$PDEOBS_DATA/smoke/summary.json" --validate-shards
pdeobs train --config configs/experiment/recovery_unet_smoke.yaml \\
  --output "$PDEOBS_RUNS/smoke-train"</code></pre>
  </div>
  <p class="muted">Use <code>pdeobs doctor --gpu</code> when CUDA is expected. The full guide also covers the strict 34-sample signal workflow, evaluation, resume, and safe Git updates.</p>
</section>

<section id="seawulf" class="section">
  <h2>SeaWulf: dependency-chained smoke example</h2>
  <p>On SeaWulf, do not build or run long work on the login node. This example pins the environment to the checked-out commit and stops downstream work automatically if validation fails.</p>
  <div class="card">
<pre><code>ssh YOUR_NETID@milan.seawulf.stonybrook.edu
module load slurm
git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git
cd PartialObs--PDEBench
git checkout YOUR_RELEASE_TAG_OR_COMMIT

export PDEOBS_GROUP=YOUR_GROUP
export PDEOBS_COMMIT="$(git rev-parse --short=12 HEAD)"
export PDEOBS_ENV="/gpfs/projects/$PDEOBS_GROUP/envs/pdeobs-$PDEOBS_COMMIT"
export PDEOBS_DATA="/gpfs/scratch/$USER/pdeobs/data"
export PDEOBS_RUNS="/gpfs/scratch/$USER/pdeobs/runs"
mkdir -p logs "$PDEOBS_DATA/plans" "$PDEOBS_RUNS"

# Build only after entering a compute allocation.
srun --partition=short-40core-shared --nodes=1 --ntasks=1 \\
  --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash -l
bash hpc/seawulf/bootstrap.sh
exit

"$PDEOBS_ENV/bin/python" -m pdeobs plan \\
  --config configs/dataset/smoke.yaml --tier tiny \\
  --output "$PDEOBS_DATA/plans/smoke.jsonl"

generation_job="$(sbatch --parsable --array=0-0 \\
  hpc/seawulf/generate_array.sbatch configs/dataset/smoke.yaml \\
  "$PDEOBS_DATA/smoke" "$PDEOBS_DATA/plans/smoke.jsonl")"
generation_job="${{generation_job%%;*}}"

validation_job="$(sbatch --parsable --dependency="afterok:$generation_job" \\
  hpc/seawulf/aggregate_cpu.sbatch "$PDEOBS_DATA/smoke" \\
  "$PDEOBS_DATA/smoke/summary.json" "$PDEOBS_DATA/plans/smoke.jsonl")"
validation_job="${{validation_job%%;*}}"

training_job="$(sbatch --parsable --dependency="afterok:$validation_job" \\
  hpc/seawulf/train_gpu.sbatch configs/experiment/recovery_unet_smoke.yaml \\
  --output "$PDEOBS_RUNS/smoke-train")"
training_job="${{training_job%%;*}}"
squeue -j "$generation_job,$validation_job,$training_job"</code></pre>
  </div>
  <div class="note"><b>Storage:</b> SeaWulf scratch is temporary and not backed up. Copy valuable validated outputs to an independent archive.</div>
</section>

<section class="section">
  <h2>Before scaling</h2>
  <ul>
    <li>Finish the smoke workflow and inspect logs, validation summaries, memory, and GPU use.</li>
    <li>Record the exact Git commit and keep resolved configurations and provenance with every run.</li>
    <li>Use the focused signal-tier example before a factorized campaign.</li>
    <li>Do not publish bundled solver outputs as paper ground truth until the numerical-validation gate passes.</li>
  </ul>
</section>
"""
    return page(
        title="Run PDE-OBS on servers",
        root=root,
        current="run",
        hero_h1="Run PDE-OBS from Git",
        hero_subtitle_html=(
            "Copy-ready paths for a single Linux server and the SeaWulf Slurm cluster."
        ),
        hero_meta_html=badges(
            ["Linux CPU/GPU", "SeaWulf Slurm", "Verified smoke first", "Exact Git revision"]
        ),
        body_html=body,
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8")


def paper_public_record(p: Dict[str, Any]) -> Dict[str, Any]:
    """Create the client-side (public) paper record written to docs/assets/papers_db.json."""
    links = p.get("links") or {}

    # Normalize common link keys (prefer explicit keys if present)
    link_pdf = links.get("pdf") or links.get("paper") or ""
    link_code = links.get("code") or links.get("repo") or ""
    link_arxiv = links.get("arxiv") or ""
    link_doi = links.get("doi") or ""
    link_openreview = links.get("openreview") or ""
    link_project = links.get("project") or ""
    links = {
        "paper": links.get("paper") or link_pdf or link_arxiv or link_doi or link_openreview,
        "pdf": link_pdf,
        "arxiv": link_arxiv,
        "doi": link_doi,
        "openreview": link_openreview,
        "code": link_code,
        "project": link_project,
    }

    pdes_list, pdes_is_auto = get_display_list(p, "pdes")
    tasks_list, tasks_is_auto = get_display_list(p, "tasks")

    is_curated = p.get("status") == "curated"

    return {
        "slug": p.get("slug"),
        "short_title": (p.get("short_title") or "").strip(),
        "full_title": (p.get("full_title") or p.get("title") or "").strip(),
        "authors": (p.get("authors") or "").strip(),
        "year": p.get("year"),
        "venue": (p.get("venue") or "").strip(),
        "method_class": (p.get("method_class") or "").strip(),
        "status": (p.get("status") or "index").strip(),
        "badges": p.get("badges", []) or [],
        "links": links,
        "pdes": pdes_list,
        "pdes_auto": bool(pdes_is_auto),
        "tasks": tasks_list,
        "tasks_auto": bool(tasks_is_auto),
        # Keep rich text fields only for curated entries (index entries remain metadata-only)
        "tldr": (p.get("tldr") or "").strip() if is_curated else "",
        "problem": (p.get("problem") or "").strip() if is_curated else "",
        "tagline": (p.get("tagline") or "").strip() if is_curated else "",
        "bibtex": (p.get("bibtex") or "").strip(),
    }


def main() -> None:
    papers = load_db()

    # Normalize lists and populate "auto" suggestions (without overwriting human fields).
    for p in papers:
        p.setdefault("method_class", infer_method_class(p))

        # --- Human-curated fields ---
        p["pdes"] = _dedup_keep_order([normalize_pde_tag(x) for x in (p.get("pdes") or [])])
        p["tasks"] = _dedup_keep_order(
            [x.strip() for x in (p.get("tasks") or []) if x and x.strip()]
        )

        # --- Auto-suggested fields (stored under p["auto"]) ---
        if not isinstance(p.get("auto"), dict):
            p["auto"] = {}
        p["auto"].setdefault("pdes", [])
        p["auto"].setdefault("tasks", [])

        # Only add suggestions if the human list is empty.
        if not p["pdes"]:
            p["auto"]["pdes"] = _dedup_keep_order(
                (p.get("auto", {}).get("pdes") or []) + infer_pdes(p)
            )
        if not p["tasks"]:
            p["auto"]["tasks"] = _dedup_keep_order(
                (p.get("auto", {}).get("tasks") or []) + infer_tasks(p)
            )

    # Write a compact JSON DB for client-side rendering
    papers_json = [paper_public_record(p) for p in papers]
    write(DOCS / "assets" / "papers_db.json", json.dumps(papers_json, ensure_ascii=False, indent=2))
    builder_options = benchmark_builder_options()
    write(
        DOCS / "assets" / "benchmark-builder-options.json",
        json.dumps(builder_options, ensure_ascii=False, indent=2),
    )
    write(
        DOCS / "assets" / "progress.json",
        json.dumps(FULL_DATASET_PROGRESS, ensure_ascii=False, indent=2),
    )

    # Core pages
    write(DOCS / "index.html", render_home(papers))
    write(DOCS / "progress" / "index.html", render_progress())
    write(DOCS / "research" / "index.html", render_research_index(papers))
    write(DOCS / "builder" / "index.html", render_builder())
    write(DOCS / "server" / "index.html", render_server())
    write(DOCS / "contribute" / "index.html", render_contribute(papers))

    # Single generic placeholder page for non-curated papers
    write(DOCS / "research" / "paper" / "index.html", render_paper_placeholder())

    # Clean per-paper directories (avoid stale pages when switching between curated/index)
    research_dir = DOCS / "research"
    if research_dir.exists():
        for child in research_dir.iterdir():
            if child.is_dir() and child.name not in {"paper"}:
                shutil.rmtree(child)

    # Curated per-paper pages only
    curated = [p for p in papers if str(p.get("status") or "index") == "curated"]
    for p in curated:
        write(DOCS / "research" / p["slug"] / "index.html", render_paper_page(p))

    # PDE problems and baselines index pages
    write(DOCS / "pde-problems" / "index.html", render_pde_problems(papers))
    write(DOCS / "baselines" / "index.html", render_baselines(papers))

    print(
        f"Generated: {len(curated)} curated paper pages + {len(papers)} index entries (papers_db.json)."
    )


if __name__ == "__main__":
    main()
