#!/usr/bin/env python3
"""Static-site generator for the *docs/* pages.

Source of truth
--------------
The generator prefers per-paper YAML files under ``data/papers/*.yaml``.
If that folder is missing/empty, it falls back to the legacy
``scripts/research_db.ndjson`` file.

It writes:
  - docs/index.html                     (homepage: summary + paper tree)
  - docs/research/index.html            (research hub + category browser)
  - docs/research/<slug>/index.html     (one page per paper)
  - docs/pde-problems/index.html        (PDE-centric index)
  - docs/baselines/index.html           (baseline-centric index)
  - docs/contribute/index.html          (how to add/curate papers)

This repo uses GitHub Pages with /docs as the site root.
"""

from __future__ import annotations

import json
import os
import shutil
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

# YAML paper database (preferred)
PAPERS_YAML_DIR = REPO_ROOT / "data" / "papers"


PAPER_TREE_ASCII = r"""
Scientific ML for PDEs (selected famous works)
├─ Physics-informed optimization
│  ├─ Deep Ritz (2018)
│  ├─ DGM / Deep Galerkin (2018)
│  ├─ DeepBSDE (2018)
│  └─ PINNs (2019)
│     ├─ cPINNs (2020)
│     ├─ SA-PINNs (2020)
│     ├─ XPINNs (2021)
│     ├─ gPINNs (2021)
│     ├─ FBPINNs (2021)
│     └─ Gradient pathology analysis (2021)
├─ Operator learning
│  ├─ DeepONet (2021)
│  ├─ Neural Operators (Kovachki et al.)
│  │  ├─ Graph Kernel Network (GKN) (2020)
│  │  └─ MGNO (2020)
│  ├─ FNO (2020)
│  │  ├─ PINO (2021)
│  │  ├─ Galerkin Transformer (2021)
│  │  ├─ WNO (2022)
│  │  │  └─ U-WNO (2024)
│  │  ├─ U-NO (2022)
│  │  └─ CNO (2023)
│  └─ Foundation / scaling
│     └─ Poseidon (2024)
├─ Diffusion / generative PDE inference
│  ├─ DiffusionPDE (2024)
│  │  ├─ FunDPS (2025)
│  │  │  └─ PRISMA (2025)
│  │  ├─ VideoPDE (2025)
│  │  └─ Conditional diffusion protocols (2024)
│  └─ Weather diffusion
│     └─ GenCast (2025)
├─ Graph simulators
│  ├─ Graph Networks for physics simulation (2020)
│  └─ MeshGraphNets (2021)
└─ Benchmarks and datasets
   ├─ PDEBench (2022)
   ├─ PDEArena (2022)
   └─ Weather: FourCastNet (2022) → Pangu-Weather (2022) → GraphCast (2023) → NeuralGCM (2023) → FengWu (2023)
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
  DiffGen --> DiffPDE["DiffusionPDE (2024)"]
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
  click DeepRitz "research/deep-ritz/" "Deep Ritz (2018)" _self
  click DGM "research/dgm/" "Deep Galerkin Method (2018)" _self
  click DeepBSDE "research/deepbsde/" "DeepBSDE (2018)" _self
  click PINN "research/pinn/" "PINNs (2019)" _self
  click cPINN "research/cpinn/" "cPINNs (2020)" _self
  click SAPINN "research/sa-pinn/" "SA-PINNs (2020)" _self
  click XPINN "research/xpinn/" "XPINNs (2021)" _self
  click gPINN "research/gpinn/" "gPINNs (2021)" _self
  click FBPINN "research/fbpinns/" "FBPINNs (2021)" _self

  click OL "research/?method=Operator%20learning" "Filter: Operator learning" _self
  click DeepONet "research/deeponet/" "DeepONet (2021)" _self
  click FNO "research/fno/" "Fourier Neural Operator (2020)" _self
  click PINO "research/pino/" "Physics-Informed Neural Operator (2021)" _self
  click GalerkinT "research/galerkin-transformer/" "Galerkin Transformer (2021)" _self
  click UNO "research/u-no/" "U-NO (2022)" _self
  click WNO "research/wno/" "WNO (2022)" _self
  click UWNO "research/u-wno/" "U-WNO (2024)" _self
  click CNO "research/cno/" "CNO (2023)" _self
  click GKN "research/gkn/" "Graph Kernel Network (2020)" _self
  click MGNO "research/mgno/" "MGNO (2020)" _self

  click DiffGen "research/?method=Diffusion" "Filter: Diffusion" _self
  click DiffPDE "research/diffusionpde/" "DiffusionPDE (2024)" _self
  click FunDPS "research/fundps/" "FunDPS (2025)" _self
  click PRISMA "research/prisma/" "PRISMA (2025)" _self
  click VideoPDE "research/videopde/" "VideoPDE (2025)" _self

  click GraphSim "research/?method=Graph%20%2F%20mesh" "Filter: Graph / mesh" _self
  click GNS "research/gns/" "GNS (ICML 2020)" _self
  click MGN "research/meshgraphnets/" "MeshGraphNets (ICLR 2021)" _self

  click Bench "benchmark/" "Benchmark tab" _self
  click PDEBench "research/pdebench/" "PDEBench (2022)" _self
  click PDEArena "research/pdearena/" "PDEArena (2022)" _self
  click FourCastNet "research/fourcastnet/" "FourCastNet (2022)" _self
  click GraphCast "research/graphcast/" "GraphCast (2023)" _self
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
        elif has_any("hemodynamics", "cfd", "turbulence") or ("fluid" in t and "fluid" not in "differential"):
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

    add("Training acceleration / stabilization", "accelerat", "fast", "efficient", "speed", "precondition", "natural gradient", "optimizer", "conflict", "loss balancing", "curriculum", "adaptive weight", "kronecker")
    add("Theory / analysis", "analysis", "convergence", "error", "generalization", "bounds", "mismatch", "failure mode", "loss landscape", "theory")
    add("Adaptive sampling / active learning", "active learning", "adaptive sampling", "residual-based", "causal sampling", "importance sampling")
    add("Uncertainty quantification", "uncertainty", "bayesian", "probabilistic", "gaussian process", "uq")

    add("Inverse problem / reconstruction", "inverse", "reconstruction", "tomography", "data assimilation", "identification", "parameter estimation", "unknown coefficient")
    add("PDE discovery / identification", "discover", "discovery", "learning pdes", "pde-net", "sindy", "model discovery", "equation")

    add("Forward prediction / rollout", "forecast", "prediction", "predicting", "rollout", "simulation", "time-dependent", "long-term")
    add("Operator learning / surrogate modeling", "operator", "neural operator", "deeponet", "fourier neural operator", "fno")

    add("Generative reconstruction / inpainting", "diffusion", "generative", "inpainting", "sampling", "posterior")
    add("Graph / mesh simulation", "graph", "mesh")
    add("Neural ODE/SDE modeling", "neural ode", "neural odes", "sde", "stochastic differential", "controlled differential")

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

    This is only a fallback when a paper YAML/DB entry does not specify
    ``method_class``.
    """
    title = str(p.get("full_title") or p.get("title") or "").lower()
    cat = str(p.get("category") or "").lower()
    text = f"{title} {cat}"

    if any(k in text for k in ["diffusion", "score-based", "score based", "denoising", "sde"]):
        return "Diffusion"
    if any(k in text for k in ["neural operator", "operator learning", "deeponet", "fno", "fourier neural operator"]):
        return "Operator learning"
    if any(k in text for k in ["pinn", "physics-informed", "physics informed", "physics-constrained", "physics constrained"]):
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
    """Load paper metadata.

    Preferred: YAML files under ``data/papers/*.yaml`` (one file per paper).
    Fallback: NDJSON under ``scripts/research_db.ndjson`` (legacy).
    """

    papers: List[Dict[str, Any]] = []

    # 1) YAML DB (preferred)
    if PAPERS_YAML_DIR.exists():
        for path in sorted(PAPERS_YAML_DIR.rglob("*.y*ml")):
            if path.name.startswith("_") or any(part.startswith("_") for part in path.parts):
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                continue
            data.setdefault("slug", path.stem)
            data["_yaml_path"] = str(path.relative_to(REPO_ROOT))

            # Allow a friendlier alias: `title:` instead of `full_title:`
            if data.get("title") and not data.get("full_title"):
                data["full_title"] = data["title"]
            if data.get("full_title") and not data.get("title"):
                data["title"] = data["full_title"]

            # Normalize common fields
            if "year" in data and isinstance(data["year"], str) and data["year"].isdigit():
                data["year"] = int(data["year"])
            data.setdefault("authors", "")
            data.setdefault("short_title", data.get("short_title") or data.get("full_title") or data["slug"])
            data.setdefault("method_class", data.get("method_class") or "SciML")
            data.setdefault("status", data.get("status") or "index")
            data.setdefault("links", {})
            data.setdefault("badges", [])
            data.setdefault("quick_facts", [])
            data.setdefault("contrib", [])
            data.setdefault("theory", [])
            data.setdefault("core_math", [])
            data.setdefault("pdes", [])
            data.setdefault("tasks", [])
            data.setdefault("baselines", [])
            data.setdefault("setting", [])
            data.setdefault("results_tables", [])
            data.setdefault("auto", {})
            if not isinstance(data["auto"], dict):
                data["auto"] = {}
            data["auto"].setdefault("pdes", [])
            data["auto"].setdefault("tasks", [])

            papers.append(data)

    if papers:
        return papers

    # 2) NDJSON DB (legacy)
    path = REPO_ROOT / "scripts" / "research_db.ndjson"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                papers.append(obj)
    return papers


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def ul(items: List[str]) -> str:
    if not items:
        return ""
    lis = "\n".join(f"<li>{i}</li>" for i in items)
    return f"<ul>\n{lis}\n</ul>"


def placeholder(text: str = "Not extracted yet.") -> str:
    return f"<p class=\"muted\">{html_escape(text)}</p>"


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
    spans = "\n".join(f"<span class=\"badge\">{html_escape(i)}</span>" for i in items)
    return f"<div class=\"badges\">\n{spans}\n</div>"


def nav(root: str, current: str) -> str:
    # current: one of "home", "research", "pde", "baselines", "benchmark", "contribute"
    def a(href: str, label: str, key: str) -> str:
        aria = ' aria-current="page"' if key == current else ""
        return f"<a href=\"{href}\"{aria}>{label}</a>"

    return (
        "<nav class=\"nav\">"
        + a(f"{root}index.html", "Home", "home")
        + a(f"{root}research/", "Research", "research")
        + a(f"{root}pde-problems/", "PDE problems", "pde")
        + a(f"{root}baselines/", "Baselines", "baselines")
        + a(f"{root}benchmark/", "Benchmark", "benchmark")
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
    now = datetime.utcnow().strftime("%Y-%m-%d")

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>{html_escape(title)}</title>
  <link rel=\"stylesheet\" href=\"{root}assets/style.css\" />
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
      <div class=\"muted\">Last generated: {now}</div>
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
            f"<div><b>Paper:</b> <a class=\"meta-link\" href=\"{links['paper']}\" target=\"_blank\" rel=\"noopener noreferrer\">{html_escape(links.get('paper_label','link'))}</a></div>"
        )
    if links.get("code"):
        meta_lines.append(
            f"<div><b>Code:</b> <a class=\"meta-link\" href=\"{links['code']}\" target=\"_blank\" rel=\"noopener noreferrer\">repository</a></div>"
        )
    if links.get("project"):
        meta_lines.append(
            f"<div><b>Project:</b> <a class=\"meta-link\" href=\"{links['project']}\" target=\"_blank\" rel=\"noopener noreferrer\">project page</a></div>"
        )

    meta_html = "\n".join(meta_lines)
    hero_meta = f"<div class=\"meta\">{meta_html}</div>" if meta_lines else ""

    quick = p.get("quick_facts", [])
    hero_card = (
        "<div class=\"hero-card\">"
        "  <div class=\"smallcaps\">Quick facts</div>"
        f"  <p class=\"muted\" style=\"margin-top:8px;\">{'<br/>'.join(quick)}</p>"
        f"  <p style=\"margin:10px 0 0;\"><a href=\"../index.html\">← Research</a> · <a href=\"../../index.html\">Home</a></p>"
        "</div>"
    )

    # --- Page body ---
    sections: List[str] = []

    # Curation status + source path
    status = str(p.get("status", "index"))
    yaml_path = p.get("_yaml_path") or f"data/papers/{p.get('slug','<slug>')}.yaml"
    if status != "curated":
        sections.append(
            "<div class=\"note\">"
            "This page is currently an <b>index-only</b> placeholder. "
            "To improve it, edit the YAML file: "
            f"<code>{html_escape(str(yaml_path))}</code> "
            "(see the <b>Contribute</b> tab)."
            "</div>"
        )

    tldr = (p.get("tldr") or "").strip()
    if not tldr:
        tldr_html = "<p class=\"muted\">Not curated yet. Add a 2–4 sentence TL;DR in the YAML file.</p>"
    else:
        tldr_html = f"<p>{html_escape(tldr)}</p>"
    sections.append(f"<section id=\"tldr\"><h2>TL;DR</h2>{tldr_html}</section>")


    # Problem statement (optional but strongly recommended for curated pages)
    problem = (p.get("problem") or "").strip()
    if not problem:
        problem_html = "<p class=\"muted\">Add <code>problem:</code> to explain what the paper is trying to solve.</p>"
    else:
        problem_html = f"<p>{html_escape(problem)}</p>"
    sections.append(f"<section id=\"problem\"><h2>Problem</h2>{problem_html}</section>")

    # Benefits vs others (optional; use bullet points)
    benefits = p.get("benefits") or p.get("advantages") or []
    if isinstance(benefits, str):
        benefits = [benefits]
    sections.append(
        "<section id=\"benefits\"><h2>Benefits vs others</h2>"
        + ul_or_placeholder(
            benefits,
            "Add <code>benefits:</code> as a bullet list (e.g., accuracy, speed, data efficiency, stability, generalization).",
        )
        + "</section>"
    )

    # Interesting notes (optional)
    interesting = p.get("interesting") or p.get("notes") or ""
    if isinstance(interesting, list):
        interesting_html = ul_or_placeholder(interesting, "Add <code>interesting:</code> as bullet points.")
    else:
        interesting = str(interesting).strip()
        interesting_html = f"<p>{html_escape(interesting)}</p>" if interesting else "<p class=\"muted\">(Optional) Add <code>interesting:</code>.</p>"
    sections.append(f"<section id=\"interesting\"><h2>Interesting detail</h2>{interesting_html}</section>")


    # Core method (math) + theory
    method_class = p.get("method_class", "SciML")
    math_lines = p.get("core_math", []) or METHOD_MATH.get(method_class, []) or METHOD_MATH["SciML"]
    sections.append(
        "<section id=\"core-math\"><h2>Core method (math)</h2>"
        f"<p class=\"muted\">Template for <b>{html_escape(method_class)}</b>. Paper-specific equations are added when manually curated.</p>"
        + (render_math_block(math_lines) if math_lines else placeholder("No template available."))
        + "</section>"
    )

    sections.append(
        "<section id=\"theory\"><h2>Main theoretical contribution</h2>"
        + ul_or_placeholder(
            p.get("theory", []),
            "Not curated yet. Add bullet points under <code>theory</code> in YAML.",
        )
        + "</section>"
    )

    sections.append(
        "<section id=\"contribution\"><h2>Main contribution</h2>"
        + ul_or_placeholder(
            p.get("contrib", []),
            "Not curated yet. Add bullet points under <code>contrib</code> in YAML.",
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
                    f"<td>{html_escape(str(r.get('metric','')))}</td>"
                    f"<td>{html_escape(str(r.get('value','')))}</td>"
                    f"<td>{html_escape(str(r.get('dataset','')))}</td>"
                    f"<td>{html_escape(str(r.get('compared_to','')))}</td>"
                    "</tr>"
                )
            main_results_html = (
                "<div class=\"tablewrap\"><table>"
                "<thead><tr><th>Metric</th><th>Value</th><th>Dataset</th><th>Compared to</th></tr></thead>"
                "<tbody>" + "".join(rows) + "</tbody></table></div>"
            )
        else:
            main_results_html = ul_or_placeholder(
                [str(x) for x in main_results],
                "Add <code>main_results</code> as a list (either dict rows or strings).",
            )
    else:
        main_results_html = "<p class=\"muted\">(Optional) Add <code>main_results</code> for a quick headline summary.</p>"

    sections.append(
        "<section id=\"main-results\"><h2>Main results (headline)</h2>"
        + main_results_html
        + "</section>"
    )

    # Experiments / PDE / tasks
    pdes_display, pdes_is_auto = get_display_list(p, "pdes")
    tasks_display, tasks_is_auto = get_display_list(p, "tasks")
    pdes_title = "PDE problems" + (" <span class=\"muted\">(auto)</span>" if pdes_is_auto else "")
    tasks_title = "Tasks" + (" <span class=\"muted\">(auto)</span>" if tasks_is_auto else "")

    exp_html = (
        "<section id=\"experiments\"><h2>Experiments</h2>"
        "<div class=\"grid2\">"
        f"  <div class=\"card\"><h3>{pdes_title}</h3>"
        + ul_or_placeholder(pdes_display, "Not specified yet.")
        + "</div>"
        f"  <div class=\"card\"><h3>{tasks_title}</h3>"
        + ul_or_placeholder(tasks_display, "Not specified yet.")
        + "</div>"
        "</div>"
    )
    exp_html += (
        "<div class=\"card\" style=\"margin-top:14px;\"><h3>Experiment setting (high level)</h3>"
        + ul_or_placeholder(p.get("setting", []))
        + "</div>"
    )
    exp_html += "</section>"
    sections.append(exp_html)

    sections.append(
        "<section id=\"baselines\"><h2>Comparable baselines</h2>"
        + ul_or_placeholder(p.get("baselines", []), "Not curated yet. Add items under <code>baselines</code> in YAML.")
        + "</section>"
    )

    # Results tables
    tables = p.get("results_tables", []) or []
    if tables:
        res_parts = ["<section id=\"results\"><h2>Main results</h2>"]
        for t in tables:
            if t.get("title"):
                res_parts.append(f"<h3 class=\"subhead\">{html_escape(t['title'])}</h3>")
            if t.get("note"):
                res_parts.append(f"<p class=\"muted\">{t['note']}</p>")
            header = t.get("header", [])
            rows = t.get("rows", [])
            thead = "".join(f"<th>{html_escape(h)}</th>" for h in header)
            body_rows = []
            for r in rows:
                body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            res_parts.append(
                "<div class=\"tablewrap\"><table><thead><tr>"
                + thead
                + "</tr></thead><tbody>"
                + "\n".join(body_rows)
                + "</tbody></table></div>"
            )
        if p.get("benchmark_note"):
            res_parts.append(f"<div class=\"note\">{p['benchmark_note']}</div>")
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
        url = links.get("paper") or links.get("arxiv") or links.get("openreview") or links.get("code") or ""
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
        "<section id=\"citation\"><h2>Citation (BibTeX)</h2>"
        + f"<pre class=\"code\"><code>{html_escape(bib)}</code></pre>"
        + "</section>"
    )



    body = "\n".join(sections)

    return page(
        title=f"{p['short_title']} ({p['year']}) — Research — PartialObs-PDEBench",
        root=root,
        current="research",
        hero_h1=f"{html_escape(p['short_title'])} ({p['year']})",
        hero_subtitle_html=f"<b>{html_escape(p['full_title'])}</b><br/>{html_escape(p.get('authors',''))}",
        hero_meta_html=hero_meta + badges(p.get("badges", [])),
        hero_card_html=hero_card,
        extra_head=(
            "<script>window.MathJax={tex:{inlineMath:[['\\(','\\)'],['$','$']]}};</script>"
            "<script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>"
            "<script defer src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>"
            "<script>document.addEventListener('DOMContentLoaded',function(){if(window.mermaid){mermaid.initialize({startOnLoad:true,theme:'dark'});}});</script>"
        ),
        body_html=body,
    )


def render_home(papers: List[Dict[str, Any]]) -> str:
    root = ""  # docs/index.html
    n_total = len(papers)
    n_curated = sum(1 for p in papers if str(p.get("status") or "index") == "curated")

    # The homepage should stay minimal (no long “get started” block).
    hero_card = (
        "<div class=\"hero-card\">"
        "  <div class=\"smallcaps\">Contribute</div>"
        "  <p class=\"muted\" style=\"margin-top:8px;\">"
        "    Add or improve paper pages by editing <code>data/papers/*.yaml</code>. "
        "    See the <b>Contribute</b> tab for the step-by-step workflow and templates."
        "  </p>"
        "</div>"
    )

    body = f"""
<section id="intro" class="section">
  <h2>What is this website?</h2>
  <p>
    PartialObs–PDEBench focuses on <b>PDE reconstruction and inference when observations are sparse</b>
    (missing sensors, masked pixels, partial trajectories).
  </p>
  <ul>
    <li><b>Research:</b> browse/search <b>{n_total}</b> AI4PDE/AI4SDE papers (<b>{n_curated}</b> curated pages + index placeholders).</li>
    <li><b>PDE problems:</b> which PDEs appear in the literature + which papers use them.</li>
    <li><b>Baselines:</b> a cross-paper index of commonly compared methods.</li>
    <li><b>Benchmark:</b> benchmark spec (PDE suite, masks, metrics, data generation) — <i>work in progress</i>.</li>
    <li><b>Contribute:</b> how to add/curate papers via YAML.</li>
  </ul>
</section>

<section id="tree" class="section">
  <h2>Paper tree</h2>
  <p class="muted">A compact, conceptual lineage map (selected famous works).</p>
  <div class="card" style="margin-top:12px;">
    <pre class="mermaid">{html_escape_pre(PAPER_TREE_MERMAID)}</pre>
  </div>
  <details style="margin-top:12px;">
    <summary class="muted">Show ASCII fallback</summary>
    <pre class="code"><code>{html_escape(PAPER_TREE_ASCII)}</code></pre>
  </details>
</section>

<section id="taxonomy" class="section">
  <h2>AI4PDE + AI4SDE map</h2>
  <p class="muted">A high-level taxonomy you can extend as new method families emerge.</p>
  <div class="card" style="margin-top:12px;">
    <pre class="mermaid">{html_escape_pre(AI4PDE_SDE_TREE_MERMAID)}</pre>
  </div>
  <details style="margin-top:12px;">
    <summary class="muted">Show ASCII fallback</summary>
    <pre class="code"><code>{html_escape(AI4PDE_SDE_TREE_ASCII)}</code></pre>
  </details>
</section>
"""

    return page(
        title="PartialObs–PDEBench",
        root=root,
        current="home",
        hero_h1="PartialObs–PDEBench",
        hero_subtitle_html=(
            "A benchmark + research map for <b>PDE inference under partial observation</b>. "
            "Homepage = summary + trees; details live in the other tabs."
        ),
        hero_meta_html=(
            "<div class=\"meta\">"
            "  <div><b>Project:</b> <a class=\"meta-link\" href=\"https://ru1ch3n.github.io/PartialObs--PDEBench\" target=\"_blank\" rel=\"noopener noreferrer\">ru1ch3n.github.io/PartialObs--PDEBench</a></div>"
            "  <div><b>Repo:</b> <a class=\"meta-link\" href=\"https://github.com/ru1ch3n/PartialObs--PDEBench\" target=\"_blank\" rel=\"noopener noreferrer\">GitHub</a></div>"
            "</div>"
        ),
        hero_card_html=hero_card,
        extra_head=(
            "<script>window.MathJax={tex:{inlineMath:[['\\(','\\)'],['$','$']]}};</script>"
            "<script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>"
            "<script defer src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>"
            "<script>document.addEventListener('DOMContentLoaded',()=>{"
            "  if(!window.mermaid) return;"
            "  // Enable clickable nodes (mermaid `click` directives) on GitHub Pages."
            "  mermaid.initialize({startOnLoad:false,theme:'dark',securityLevel:'loose'});"
            "  mermaid.run({querySelector:'.mermaid'});"
            "});</script>"
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
  <h2>Research index</h2>
  <p class="muted">
    Browse/search <b>{n_total}</b> papers. Curated pages include structured experiment + baseline notes;
    index pages are bibliographic placeholders.
  </p>

  <div class="card" style="margin-top:16px;">
    <div class="grid">
      <div>
        <div class="smallcaps">Search</div>
        <input id="q" class="input" placeholder="Search title / authors / venue..." />
      </div>

      <div>
        <div class="smallcaps">Filters</div>
        <div class="row">
          <select id="f_method" class="select"><option value="">All methods</option></select>
          <select id="f_venue" class="select"><option value="">All venues</option></select>
          <select id="f_pde" class="select"><option value="">All PDEs</option></select>
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
    Tip: if you find an index placeholder you care about, click into it and use the <b>Contribute</b> tab to add a curated YAML summary.
  </div>
</section>
"""

    return page(
        title="Research index",
        root=root,
        current="research",
        hero_h1="Research index",
        hero_subtitle_html="Browse and search AI4PDE/AI4SDE papers.",
        hero_meta_html="",
        hero_card_html="",
        extra_head=(
            "<script>window.PAPERS_DB_URL='../assets/papers_db.json';</script>"
            "<script defer src=\"../assets/research.js\"></script>"
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
            "<script defer src=\"../../assets/paper.js\"></script>"
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
                f"<td class=\"muted\">{html_escape(top_methods) if top_methods else '—'}</td>"
                f"<td class=\"muted\">{html_escape(top_bases) if top_bases else '—'}</td>"
                f"<td><a href=\"{link}\">View papers</a></td>"
                "</tr>"
            )

        sections.append(
            "<section class=\"section\">"
            f"  <h2>{html_escape(fam)}</h2>"
            "  <div class=\"tablewrap\"><table><thead><tr>"
            "    <th>PDE</th><th># papers</th><th>Common method classes</th><th>Common baselines (curated pages)</th><th></th>"
            "  </tr></thead><tbody>"
            + "\n".join(rows)
            + "</tbody></table></div>"
            "</section>"
        )

    body = (
        "<section class=\"section\">"
        "  <h2>Browse by PDE problem</h2>"
        "  <p>This page groups PDEs into common families. Each row links to the Research table with a PDE filter applied.</p>"
        "  <div class=\"note\">PDE tags for <b>index-only</b> papers are auto-extracted from titles and may be incomplete. Curated pages include better coverage.</div>"
        "</section>"
        + "\n".join(sections)
    )

    hero_card = (
        "<div class=\"hero-card\">"
        "  <div class=\"smallcaps\">How to use</div>"
        "  <p class=\"muted\" style=\"margin-top:8px;\">"
        "    Pick a PDE family → click <b>View papers</b> to jump into the Research table."
        "  </p>"
        "</div>"
    )

    return page(
        title="PDE problems — PartialObs–PDEBench",
        root=root,
        current="pde",
        hero_h1="PDE problems",
        hero_subtitle_html="Group PDEs by family and see which methods/baselines appear across papers.",
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
            f"<td><a href=\"{link}\">View papers</a></td>"
            "</tr>"
        )
    cls_table = (
        "<div class=\"tablewrap\"><table><thead><tr>"
        "<th>Method class</th><th># papers</th><th>Core objective (template)</th><th></th>"
        "</tr></thead><tbody>"
        + "\n".join(cls_rows)
        + "</tbody></table></div>"
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
            f"<a href=\"../research/{pp['slug']}/\">{html_escape(pp['short_title'])}</a>" for pp in ex
        )
        qlink = f"../research/?q={quote(base)}"
        base_rows.append(
            "<tr>"
            f"<td><b>{html_escape(base)}</b></td>"
            f"<td>{len(ps)}</td>"
            f"<td class=\"muted\">{ex_links or '—'}</td>"
            f"<td><a href=\"{qlink}\">Search</a></td>"
            "</tr>"
        )
    base_table = (
        "<div class=\"tablewrap\"><table><thead><tr>"
        "<th>Baseline method</th><th># curated papers</th><th>Examples</th><th></th>"
        "</tr></thead><tbody>"
        + "\n".join(base_rows)
        + "</tbody></table></div>"
    )

    body = (
        "<section class=\"section\">"
        "  <h2>Method classes</h2>"
        "  <p>High-level taxonomy used across this website. Each row links to the Research table with a class filter.</p>"
        "</section>"
        + cls_table
        + "<section class=\"section\">"
        "  <h2>Baseline methods (curated pages)</h2>"
        "  <p>Baseline lists are only available on manually curated paper pages. This table summarizes what is currently extracted.</p>"
        "</section>"
        + base_table
    )

    hero_card = (
        "<div class=\"hero-card\">"
        "  <div class=\"smallcaps\">Tip</div>"
        "  <p class=\"muted\" style=\"margin-top:8px;\">"
        "    Use <b>Method classes</b> to compare approaches, and <b>Baseline methods</b> to reproduce literature tables." 
        "  </p>"
        "</div>"
    )

    return page(
        title="Baselines — PartialObs–PDEBench",
        root=root,
        current="baselines",
        hero_h1="Baselines & method taxonomy",
        hero_subtitle_html="Tables of method classes and commonly compared baselines.",
        hero_card_html=hero_card,
        extra_head=(
            "<script>window.MathJax={tex:{inlineMath:[['\\(','\\)'],['$','$']]}};</script>"
            "<script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>"
        ),
        body_html=body,
    )


def render_contribute(papers: List[Dict[str, Any]]) -> str:
    root = "../"  # docs/contribute/index.html
    n_total = len(papers)
    n_curated = sum(1 for p in papers if str(p.get("status") or "index") == "curated")

    example_yaml = """slug: my-paper-2025
status: index   # or: curated
category: AI4PDE
method_class: NeuralOperator

full_title: Full paper title goes here
short_title: MyPaper
authors: First Author; Second Author; ...
year: 2025
venue: ICLR

links:
  paper: https://arxiv.org/abs/xxxx.xxxxx
  code: https://github.com/user/repo

tldr: >-
  2–4 sentences: setting → method → key result (include dataset/metric if possible).

problem: >-
  What problem does this paper solve? Be concrete (PDE, observation type, task).

contrib:
  - Main contribution #1
  - Main contribution #2

benefits:
  - What is better compared to common baselines?

baselines:
  - FNO
  - PINO

main_results:
  - metric: relative L2 error
    value: 0.012
    dataset: PDEBench Navier–Stokes
    compared_to: FNO

interesting: >-
  Any notable detail (failure mode, trick, surprising ablation, limitation).

bibkey: MyPaper2025
bibtex: |
  @inproceedings{MyPaper2025,
    title={...},
    author={...},
    booktitle={ICLR},
    year={2025},
    url={https://...}
  }
"""

    bulk_cmd = """# 1) Download a conference BibTeX file (ICLR/ICML/NeurIPS).
# 2) Convert it into YAML stubs (status=index) under data/papers/import/...

python scripts/import_bibtex_to_yaml.py \
  --bib path/to/ICLR_2025.bib \
  --venue ICLR \
  --out data/papers/import/iclr2025 \
  --status index

# Optional: only keep likely AI4PDE/AI4SDE papers by title keywords
python scripts/import_bibtex_to_yaml.py \
  --bib path/to/ICLR_2025.bib \
  --venue ICLR \
  --out data/papers/import/iclr2025 \
  --status index \
  --keywords "pde,operator,neural operator,physics-informed,navier,burgers,sde,diffusion"
"""

    build_cmd = """# Validate YAML (fast)
python scripts/validate_papers.py

# Regenerate docs/ (GitHub Pages output)
python scripts/generate_research_site.py
"""

    submit_cmd = """# After regenerating docs/, commit and push.

git status
git add -A
git commit -m "Update papers and regenerate site"
git push origin main

# If Git complains about missing user identity:
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
"""

    export_bib = """1) Go to the Research tab.
2) Tick the “Pick” checkbox for the papers you want.
3) Click “Download .bib” (or “Copy BibTeX”).
"""

    body = f"""
<section class="section">
  <h2>Contribute</h2>
  <p>
    The site is <b>YAML-first</b>: paper metadata + summaries live in <code>data/papers/</code>,
    and the website under <code>docs/</code> is generated from those YAML files.
  </p>
  <ul>
    <li><b>Index placeholders</b> (status=<code>index</code>) are quick to add: title/authors/venue/links.</li>
    <li><b>Curated pages</b> (status=<code>curated</code>) include structured notes: contributions, baselines, results, etc.</li>
  </ul>
  <div class="note">
    Current DB: <b>{n_total}</b> papers (<b>{n_curated}</b> curated).
  </div>
</section>

<section class="section">
  <h2>Add a paper (one-by-one)</h2>
  <ol>
    <li>Copy <code>data/papers/_template.yaml</code> → create <code>data/papers/&lt;slug&gt;.yaml</code>.</li>
    <li>Fill the required metadata: title, authors, year, venue, links.</li>
    <li>Start with <code>status: index</code>. Later upgrade to <code>status: curated</code> with more fields.</li>
    <li>Run validation + site generation:</li>
  </ol>
  <pre class="code"><code>{html_escape(build_cmd)}</code></pre>
</section>

<section class="section">
  <h2>Submit to GitHub (so the website updates)</h2>
  <p>
    GitHub Pages serves this website from the <code>docs/</code> folder.
    After you edit YAML files, make sure you regenerate the site, then commit + push.
  </p>
  <pre class="code"><code>{html_escape(submit_cmd)}</code></pre>
  <div class="note">
    Tip: if you are contributing via a pull request, do these steps on a feature branch and then open a PR.
  </div>
</section>

<section class="section">
  <h2>YAML format</h2>
  <p>
    Minimal fields are enough for an index entry. For curated pages, please also fill
    <code>tldr</code>, <code>problem</code>, <code>contrib</code>, <code>baselines</code>, and <code>main_results</code>.
  </p>
  <pre class="code"><code>{html_escape(example_yaml)}</code></pre>
</section>

<section class="section">
  <h2>Bulk import from conferences (ICLR / ICML / NeurIPS)</h2>
  <p>
    If you want to include <b>many papers</b> (e.g., all accepted papers for the last 5 years),
    the recommended workflow is:
  </p>
  <ol>
    <li>Get a <b>BibTeX</b> file for the conference/year.</li>
    <li>Convert BibTeX → YAML stubs with <code>scripts/import_bibtex_to_yaml.py</code>.</li>
    <li>Curate a subset (set <code>status: curated</code> + fill the summary fields) over time.</li>
  </ol>
  <pre class="code"><code>{html_escape(bulk_cmd)}</code></pre>
  <div class="note">
    Tip: importing <i>all</i> papers from top conferences can produce a very large DB.
    For AI4PDE focus, start with keyword-filtered imports and gradually expand.
  </div>
</section>

<section class="section">
  <h2>Export BibTeX from the website</h2>
  <p>
    The Research page supports batch BibTeX export:
  </p>
  <pre class="code"><code>{html_escape(export_bib)}</code></pre>
  <p class="muted">
    If a paper YAML includes <code>bibtex:</code>, we export that. Otherwise we generate a minimal BibTeX entry from metadata.
  </p>
</section>

<section class="section">
  <h2>Quality bar for curated pages</h2>
  <ul>
    <li><b>Main contribution:</b> 2–5 bullets that are specific (not marketing language).</li>
    <li><b>Problem & setting:</b> PDE(s), observation type (masked pixels / sparse sensors / partial trajectories), task (forecasting / inverse / reconstruction).</li>
    <li><b>Baselines:</b> methods actually compared in the paper (FNO/PINO/UNet/etc.).</li>
    <li><b>Main results:</b> include metric + dataset + comparison target (even 1–2 rows are useful).</li>
  </ul>
</section>
"""

    return page(
        title="Contribute",
        root=root,
        current="contribute",
        hero_h1="Contribute",
        hero_subtitle_html="How to add and curate papers (YAML-first).",
        hero_meta_html="",
        hero_card_html="",
        body_html=body,
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def paper_public_record(p: Dict[str, Any]) -> Dict[str, Any]:
    """Compact JSON record used by the client-side Research UI.

    Keep this reasonably small: it will be shipped to browsers as papers_db.json.
    """
    links = p.get("links", {}) or {}
    bib = ""
    if isinstance(p.get("bib"), dict):
        bib = (p.get("bib") or {}).get("entry", "") or ""
    bib = (p.get("bibtex") or bib or "").strip()

    return {
        "slug": p.get("slug", ""),
        "short_title": p.get("short_title", p.get("slug", "")),
        "full_title": p.get("full_title", p.get("title", "")),
        "authors": p.get("authors", ""),
        "year": (int(p.get("year")) if str(p.get("year","")).isdigit() else 0),
        "venue": p.get("venue", ""),
        "category": p.get("category", "AI4PDE"),
        "method_class": p.get("method_class", "SciML"),
        "status": str(p.get("status") or "index"),
        "badges": p.get("badges", []) or [],
        "links": links,
        "pdes": get_display_list(p, "pdes"),
        "tasks": get_display_list(p, "tasks"),
        "tldr": (p.get("tldr") or "").strip(),
        "problem": (p.get("problem") or "").strip(),
        "benefits": p.get("benefits", []) or [],
        "bibkey": p.get("bibkey", ""),
        "bibtex": bib,
    }


def main() -> None:
    papers = load_db()

    # Normalize lists and populate "auto" suggestions (without overwriting human fields).
    for p in papers:
        p.setdefault("method_class", infer_method_class(p))

        # --- Human-curated fields ---
        p["pdes"] = _dedup_keep_order([normalize_pde_tag(x) for x in (p.get("pdes") or [])])
        p["tasks"] = _dedup_keep_order([x.strip() for x in (p.get("tasks") or []) if x and x.strip()])

        # --- Auto-suggested fields (stored under p["auto"]) ---
        if not isinstance(p.get("auto"), dict):
            p["auto"] = {}
        p["auto"].setdefault("pdes", [])
        p["auto"].setdefault("tasks", [])

        # Only add suggestions if the human list is empty.
        if not p["pdes"]:
            p["auto"]["pdes"] = infer_pdes(p)
        if not p["tasks"]:
            p["auto"]["tasks"] = infer_tasks(p)

    # Write a compact JSON DB for client-side rendering
    papers_json = [paper_public_record(p) for p in papers]
    write(DOCS / "assets" / "papers_db.json", json.dumps(papers_json, ensure_ascii=False, indent=2))

    # Core pages
    write(DOCS / "index.html", render_home(papers))
    write(DOCS / "research" / "index.html", render_research_index(papers))
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

    print(f"Generated: {len(curated)} curated paper pages + {len(papers)} index entries (papers_db.json).")


if __name__ == "__main__":
    main()