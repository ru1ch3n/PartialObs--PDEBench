# PartialObs–PDEBench
**A benchmark + research map for PDE reconstruction and inference under partial observation.**

PartialObs–PDEBench studies learning-based reconstruction of PDE solution fields when observations are incomplete (e.g., **missing sensors**, **masked pixels**, or **partial trajectories**). The repository couples:

- a **research index** (GitHub Pages) that organizes AI4PDE/AI4SDE papers and curated one-page summaries, and  
- a **benchmark harness** (work in progress) for standardized partial-observation evaluations.

**Project website:** https://ru1ch3n.github.io/PartialObs--PDEBench/  
**Repository:** https://github.com/ru1ch3n/PartialObs--PDEBench

---

## Scope and problem setting

We consider a (possibly parametric) PDE with solution field \(u\). Under partial observation, we observe

\[
y = \mathcal{M}(u) + \epsilon,
\]

where \(\mathcal{M}\) is an observation operator (mask / sampling pattern / sensor map) and \(\epsilon\) is optional noise.

**Goal:** reconstruct the full field \(u\) (or predict future states) from \((y, \mathcal{M})\).

---

## What is included (today)

### 1) Website: research map (stable)
The GitHub Pages site lives in `docs/` and provides:

- **Research index:** browse/search papers; most entries are **index-only** bibliographic records.
- **Curated pages (high standard):** detailed, human-written summaries that include:
  - main idea and method formulation (math when appropriate),
  - PDE settings and tasks,
  - experimental setup + baselines,
  - main results (tables/figures when available).

> Current curated set (seed): **FNO**, **DeepONet**, **DiffusionPDE**.

### 2) Benchmark harness (work in progress)
The benchmark tab on the website (and the corresponding code/configs here) is under active development:
- PDE suite definitions, mask operators, metrics, and reproducible configs will be expanded and stabilized over time.

---

## Repository structure

- `docs/` — static website (served by GitHub Pages).
- `scripts/` — site generation utilities.
- `scripts/research_db.ndjson` — **paper index** (NDJSON; metadata-only entries).
- `data/curations/*.json` — **curated paper pages** (one JSON per curated paper).
- `configs/` — benchmark config stubs (WIP).

---

## Contributing (research index + curated pages)

The intended workflow is PR-friendly:

1. **Add an index entry** to `scripts/research_db.ndjson` (title/authors/venue/year/links/tags).  
2. Optionally **add a curated JSON** to `data/curations/<slug>.json`.  
3. Rebuild the static site:
   ```bash
   python scripts/generate_research_site.py
   ```
4. Commit the regenerated `docs/` outputs.

The website’s **Contribute** tab documents the JSON schema and review standard.

---

## Local preview

```bash
python -m http.server 8000 --directory docs
```

Open:
```text
http://localhost:8000
```

---

## Citation

If you use this repository or the website in academic work, please cite:

```bibtex
@misc{partialobs_pdebench,
  title        = {PartialObs--PDEBench: PDE reconstruction under partial observation},
  author       = {Ruichen and contributors},
  howpublished = {GitHub repository},
  year         = {2026},
  url          = {https://github.com/ru1ch3n/PartialObs--PDEBench}
}
```

---

## License
- Repository code is released under the **MIT License** (see `LICENSE`).
- External datasets / pretrained weights / upstream code remain under their respective licenses (see `THIRD_PARTY_NOTICES.md`).
