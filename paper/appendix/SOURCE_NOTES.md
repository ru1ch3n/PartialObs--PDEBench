# Source and claim notes

## Structural reference

The four-section organization follows the supplementary-material pattern in:

- PDEBench: An Extensive Benchmark for Scientific Machine Learning,
  arXiv:2210.07182.

Only the organizational pattern was used. The equations, solver descriptions,
tables, plots, and wording in this artifact were written for PDE-OBS.

## Project sources

- Benchmark factors, split and evaluation contract: `docs/BENCHMARK_PAPER.md`.
- Numerical routing and parameters: `src/pdeobs/pdes/` and
  `configs/dataset/numerics_full_t15.yaml`.
- Observation protocols: `src/pdeobs/masks.py`.
- Metric definitions: `src/pdeobs/metrics.py`.
- Data schema and provenance: `src/pdeobs/schema.py`,
  `src/pdeobs/generation.py`, and `src/pdeobs/quality.py`.
- Software and SeaWulf workflow: `pyproject.toml`,
  `environment-generation.yml`, and `hpc/seawulf/README.md`.
- Quantitative solver preflight: `release/NUMERICS_VALIDATION20.md`.
- Full-tier figure snapshot: `figure_data/figure_snapshot.json`, rendered
  read-only on SeaWulf by `paper/iclr2027/scripts/render_seawulf_figures.py`.

## Interpretation boundary

The validation20 loss values are an operator-matched implementation preflight,
not an independent discretization-convergence certificate and not a learned
model result. The full-T15 quality figure includes only completed, accepted
shards at its recorded rendering time; it is not the final frozen-plan strict
aggregate and excludes rejected or absent shards. The full T15 campaign stores
15 temporal frames per dynamic sample. Observation masks are applied only
after a ground-truth field has been generated. No model training was performed
to create this appendix.
