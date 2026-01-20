#!/usr/bin/env python3
"""Static-site generator for the *docs/* pages.

It reads paper metadata from scripts/research_db.ndjson and writes:
  - docs/index.html                     (homepage: summary + paper tree)
  - docs/research/index.html            (research hub + category browser)
  - docs/research/<slug>/index.html     (one page per paper)
  - docs/pde-problems/index.html        (PDE-centric index)
  - docs/baselines/index.html           (baseline-centric index)

This repo uses GitHub Pages with /docs as the site root.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"


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
graph TD
  Root[Scientific ML for PDEs]
  Root --> PI[Physics-informed optimization]
  PI --> DeepRitz[Deep Ritz (2018)]
  PI --> DGM[DGM / Deep Galerkin (2018)]
  PI --> DeepBSDE[DeepBSDE (2018)]
  PI --> PINN[PINNs (2019)]
  PINN --> cPINN[cPINNs (2020)]
  PINN --> SAPINN[SA-PINNs (2020)]
  PINN --> XPINN[XPINNs (2021)]
  PINN --> gPINN[gPINNs (2021)]
  PINN --> FBPINN[FBPINNs (2021)]
  Root --> OL[Operator learning]
  OL --> DeepONet[DeepONet (2021)]
  OL --> FNO[FNO (2020)]
  FNO --> PINO[PINO (2021)]
  FNO --> GalerkinT[Galerkin Transformer (2021)]
  FNO --> WNO[WNO (2022)]
  WNO --> UWNO[U-WNO (2024)]
  FNO --> UNO[U-NO (2022)]
  FNO --> CNO[CNO (2023)]
  OL --> GKN[GKN (2020)]
  OL --> MGNO[MGNO (2020)]
  Root --> Diff[Diffusion / generative PDE inference]
  Diff --> DiffPDE[DiffusionPDE (2024)]
  DiffPDE --> FunDPS[FunDPS (2025)]
  FunDPS --> PRISMA[PRISMA (2025)]
  DiffPDE --> VideoPDE[VideoPDE (2025)]
  Root --> Graph[Graph simulators]
  Graph --> GNS[Graph Networks for physics simulation (2020)]
  Graph --> MGN[MeshGraphNets (2021)]
  Root --> Bench[Benchmarks and datasets]
  Bench --> PDEBench[PDEBench (2022)]
  Bench --> PDEArena[PDEArena (2022)]
  Bench --> FourCastNet[FourCastNet (2022)]
  FourCastNet --> GraphCast[GraphCast (2023)]
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

def load_db() -> List[Dict[str, Any]]:
    """Load paper metadata.

    We use NDJSON (one JSON object per line) so the database can be edited/appended
    without reformatting a huge JSON array.
    """
    path = REPO_ROOT / "scripts" / "research_db.ndjson"
    papers: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            assert isinstance(obj, dict)
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
    # current: one of "home", "research", "pde", "baselines", "benchmark"
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
    return f"{root_to_docs}research/{p['slug']}/"


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

    sections = []
    sections.append(
        f"<section id=\"tldr\"><h2>TL;DR</h2><p>{p.get('tldr','')}</p></section>"
    )

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
        + ul_or_placeholder(p.get("theory", []), "Not extracted yet (paper page is index-only, or theory not summarized).")
        + "</section>"
    )

    sections.append(
        "<section id=\"contribution\"><h2>Main contribution</h2>"
        + ul_or_placeholder(p.get("contrib", []), "Not extracted yet (paper page is index-only).")
        + "</section>"
    )

    # Experiments / PDE / tasks
    exp_html = (
        "<section id=\"experiments\"><h2>Experiments</h2>"
        "<div class=\"grid2\">"
        "  <div class=\"card\"><h3>PDE problems (as reported)</h3>"
        + ul_or_placeholder(p.get("pdes", []), "Not specified in the source list entry.")
        + "</div>"
        "  <div class=\"card\"><h3>Tasks</h3>"
        + ul_or_placeholder(p.get("tasks", []), "Not specified in the source list entry.")
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
        + ul_or_placeholder(p.get("baselines", []))
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
    # Homepage stays compact: only website intro + a paper tree.

    hero_card = (
        "<div class=\"hero-card\">"
        "  <div class=\"smallcaps\">Quick start</div>"
        "  <p class=\"muted\" style=\"margin-top:8px;\">"
        "    <b>Goal:</b> reconstruct PDE solutions under partial observation.<br/>"
        "    <b>Browse:</b> Research → paper pages; PDE problems; Baselines.<br/>"
        "    <b>Benchmark:</b> moved to the Benchmark tab." 
        "  </p>"
        "</div>"
    )

    body = f"""
<section id=\"intro\" class=\"section\">
  <h2>What is this website?</h2>
  <p>
    PartialObs–PDEBench focuses on <b>PDE reconstruction and inference when observations are sparse</b>
    (missing sensors, masked pixels, partial trajectories). The site is organized as:
  </p>
  <ul>
    <li><b>Research:</b> an index of ~300 AI4PDE papers with hyperlinks to one-page summaries (curated pages include structured experiment tables).</li>
    <li><b>PDE problems:</b> which PDEs appear in the literature + which papers use them.</li>
    <li><b>Baselines:</b> a cross-paper index of commonly compared methods.</li>
    <li><b>Benchmark:</b> the benchmark specification (PDE suite, masks, metrics, data generation).</li>
  </ul>
</section>

<section id=\"tree\" class=\"section\">
  <h2>Paper tree</h2>
  <p class=\"muted\">A compact, conceptual lineage map for famous SciML-for-PDE works.</p>
  <div class=\"card\" style=\"margin-top:12px;\">
    <pre class=\"mermaid\">{html_escape_pre(PAPER_TREE_MERMAID)}</pre>
  </div>
  <details style=\"margin-top:12px;\">
    <summary class=\"muted\">Show ASCII fallback</summary>
    <pre class=\"code\"><code>{html_escape(PAPER_TREE_ASCII)}</code></pre>
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
            "Homepage = summary + tree; details live in the other tabs."
        ),
        hero_meta_html=(
            "<div class=\"meta\">"
            "  <div><b>Project:</b> <a class=\"meta-link\" href=\"https://ru1ch3n.github.io/PartialObs--PDEBench/\" target=\"_blank\" rel=\"noopener noreferrer\">ru1ch3n.github.io/PartialObs--PDEBench</a></div>"
            "  <div><b>Repo:</b> <a class=\"meta-link\" href=\"https://github.com/ru1ch3n/PartialObs--PDEBench\" target=\"_blank\" rel=\"noopener noreferrer\">GitHub</a></div>"
            "</div>"
        ),
        hero_card_html=hero_card,
        extra_head=(
            "<script>window.MathJax={tex:{inlineMath:[['\\(','\\)'],['$','$']]}};</script>"
            "<script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>"
            "<script defer src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>"
            "<script>document.addEventListener('DOMContentLoaded',()=>{if(window.mermaid){mermaid.initialize({startOnLoad:true,theme:'dark'});}});</script>"
        ),
        body_html=body,
    )


def render_research_index(papers: List[Dict[str, Any]]) -> str:
    root = "../"  # from docs/research/index.html
    # Filters
    method_classes = sorted({p.get("method_class", "SciML") for p in papers})
    pde_tags = sorted({pde for p in papers for pde in (p.get("pdes", []) or [])})
    statuses = sorted({p.get("status", "curated") for p in papers})

    def opt(value: str) -> str:
        return f"<option value=\"{html_escape(value)}\">{html_escape(value)}</option>"

    method_opts = "".join([opt(m) for m in method_classes])
    pde_opts = "".join([opt(p) for p in pde_tags])
    status_opts = "".join([opt(s) for s in statuses])

    # Table rows
    rows = []
    for p in sorted(papers, key=lambda x: (-(x.get("year", 0)), x.get("short_title", ""))):
        title = html_escape(p.get("short_title", ""))
        full = html_escape(p.get("full_title", ""))
        year = int(p.get("year", 0) or 0)
        method = html_escape(p.get("method_class", "SciML"))
        status = html_escape(p.get("status", "curated"))
        pdes = p.get("pdes", []) or []
        pde_str = ", ".join(html_escape(x) for x in pdes[:3]) + ("…" if len(pdes) > 3 else "")
        tasks = p.get("tasks", []) or []
        task_str = ", ".join(html_escape(x) for x in tasks[:2]) + ("…" if len(tasks) > 2 else "")
        rows.append(
            f"<tr class=\"paper-row\" "
            f"data-title=\"{full.lower()}\" "
            f"data-method=\"{method.lower()}\" "
            f"data-pdes=\"{','.join([x.lower() for x in pdes])}\" "
            f"data-status=\"{status.lower()}\">"
            f"<td>{year}</td>"
            f"<td><a href=\"{p['slug']}/\"><b>{title}</b></a><div class=\"muted\" style=\"font-size:12px;\">{full}</div></td>"
            f"<td>{method}</td>"
            f"<td class=\"muted\">{pde_str or 'Unspecified'}</td>"
            f"<td class=\"muted\">{task_str or 'Unspecified'}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )

    table_html = (
        "<div class=\"tablewrap\">"
        "<table id=\"papers\"><thead><tr>"
        "<th>Year</th><th>Paper</th><th>Method</th><th>PDEs</th><th>Tasks</th><th>Status</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )

    body = (
        "<section class=\"section\">"
        "  <h2>Research index</h2>"
        "  <p>Browse and search ~300 recent AI4PDE papers. <b>Curated</b> pages include structured experiment + baseline notes; <b>index</b> pages are bibliographic placeholders.</p>"
        "  <div class=\"card\" style=\"margin-top:14px;\">"
        "    <div class=\"grid2\">"
        "      <div>"
        "        <label class=\"smallcaps\">Search</label><br/>"
        "        <input id=\"q\" class=\"input\" placeholder=\"Search title…\" />"
        "      </div>"
        "      <div>"
        "        <label class=\"smallcaps\">Filters</label><br/>"
        "        <div class=\"filters\">"
        f"          <select id=\"method\" class=\"select\"><option value=\"\">All methods</option>{method_opts}</select>"
        f"          <select id=\"pde\" class=\"select\"><option value=\"\">All PDEs</option>{pde_opts}</select>"
        f"          <select id=\"status\" class=\"select\"><option value=\"\">All statuses</option>{status_opts}</select>"
        "        </div>"
        "      </div>"
        "    </div>"
        "    <div class=\"muted\" style=\"margin-top:10px;\"><span id=\"count\"></span></div>"
        "  </div>"
        "</section>"
        + table_html
    )

    extra_head = (
        "<style>.filters{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}.input,.select{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:rgba(15,21,34,.6);color:var(--fg)}.select{width:auto}</style>"
        "<script>\n"
        "function applyFilters(){\n"
        "  const q=(document.getElementById('q').value||'').toLowerCase();\n"
        "  const m=(document.getElementById('method').value||'').toLowerCase();\n"
        "  const p=(document.getElementById('pde').value||'').toLowerCase();\n"
        "  const s=(document.getElementById('status').value||'').toLowerCase();\n"
        "  let shown=0;\n"
        "  document.querySelectorAll('tr.paper-row').forEach(tr=>{\n"
        "    const okQ=!q || tr.dataset.title.includes(q);\n"
        "    const okM=!m || tr.dataset.method===m;\n"
        "    const okP=!p || (tr.dataset.pdes||'').includes(p);\n"
        "    const okS=!s || tr.dataset.status===s;\n"
        "    const show=okQ && okM && okP && okS;\n"
        "    tr.style.display=show?'':'none';\n"
        "    if(show) shown++;\n"
        "  });\n"
        "  document.getElementById('count').textContent = `Showing ${shown} / ${document.querySelectorAll('tr.paper-row').length}`;\n"
        "}\n"
        "document.addEventListener('DOMContentLoaded',()=>{\n"
        "  const params=new URLSearchParams(window.location.search);\n"
        "  if(params.get('q')) document.getElementById('q').value=params.get('q');\n"
        "  if(params.get('method')) document.getElementById('method').value=params.get('method');\n"
        "  if(params.get('pde')) document.getElementById('pde').value=params.get('pde');\n"
        "  if(params.get('status')) document.getElementById('status').value=params.get('status');\n"
        "  ['q','method','pde','status'].forEach(id=>document.getElementById(id).addEventListener('input',applyFilters));\n"
        "  ['method','pde','status'].forEach(id=>document.getElementById(id).addEventListener('change',applyFilters));\n"
        "  applyFilters();\n"
        "});\n"
        "</script>"
    )

    hero_card = (
        "<div class=\"hero-card\">"
        "  <div class=\"smallcaps\">At a glance</div>"
        f"  <p class=\"muted\" style=\"margin-top:8px;\">\n"
        f"    <b>Total papers:</b> {len(papers)}<br/>\n"
        "    <b>Tip:</b> use filters; or jump in via PDE problems / Baselines tabs.\n"
        "  </p>"
        "  <p style=\"margin: 10px 0 0;\"><a href=\"../index.html\">← Home</a></p>"
        "</div>"
    )

    return page(
        title="Research — PartialObs–PDEBench",
        root=root,
        current="research",
        hero_h1="Research",
        hero_subtitle_html="A searchable index of AI4PDE papers with one-page summaries.",
        hero_meta_html=(
            "<div class=\"meta\"><div><b>Legend:</b> <b>curated</b> = structured experiments/tables; <b>index</b> = bibliographic placeholder awaiting curation.</div></div>"
        ),
        hero_card_html=hero_card,
        extra_head=extra_head,
        body_html=body,
    )


def render_pde_problems(papers: List[Dict[str, Any]]) -> str:
    root = "../"  # docs/pde-problems/index.html
    pde_to_papers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in papers:
        for pde in p.get("pdes", []) or []:
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


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    papers = load_db()

    # Auto-fill missing PDE/problem + task tags (primarily for index-only entries).
    for p in papers:
        # Normalize / infer PDE tags
        p.setdefault("pdes", [])
        if p.get("pdes"):
            p["pdes"] = _dedup_keep_order([normalize_pde_tag(x) for x in (p.get("pdes") or [])])
        else:
            p["pdes"] = infer_pdes(p)

        # Normalize / infer task tags
        p.setdefault("tasks", [])
        if p.get("tasks"):
            p["tasks"] = _dedup_keep_order([x.strip() for x in (p.get("tasks") or []) if x and x.strip()])
        else:
            p["tasks"] = infer_tasks(p)

    # Home
    write(DOCS / "index.html", render_home(papers))

    # Research index
    write(DOCS / "research" / "index.html", render_research_index(papers))

    # Per-paper pages
    for p in papers:
        write(DOCS / "research" / p["slug"] / "index.html", render_paper_page(p))

    # PDE problems and baselines index pages
    write(DOCS / "pde-problems" / "index.html", render_pde_problems(papers))
    write(DOCS / "baselines" / "index.html", render_baselines(papers))

    print(f"Generated {len(papers)} paper pages.")


if __name__ == "__main__":
    main()
