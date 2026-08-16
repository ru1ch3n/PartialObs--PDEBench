# PDE-OBS benchmark-paper contract

The only manuscript in scope for this repository is:

> **PDE-OBS: A Controlled Partial-Observation Benchmark for PDE Dynamics**

Its central question is: **How should we evaluate PDE learning under partial
observations in a controlled, factorized, and reproducible way?**

The contribution is the dataset design, task suite, official splits, metrics,
anchor leaderboard, difficulty analysis, and one-line tools. It is not a paper
claiming a new semantic-ID method, a new large world model, or a new foundation
model. Those are separate projects.

The machine-readable source of truth is `pdeobs.protocol`. Check it against the
default data design with:

```bash
pdeobs protocol --check
pdeobs protocol --json
```

## Dataset design

The frozen factorization is:

```text
PDE family -> boundary protocol -> setting method -> parameter regime -> instance
```

It contains seven PDE families, four boundary protocols, ten settings, and
three regimes:

- PDEs: Darcy, Poisson, Helmholtz, heat, reaction-diffusion, Burgers, and
  Navier–Stokes.
- Boundaries: Dirichlet/no-slip, Neumann/free-slip, periodic, and
  Robin/mixed/obstacle.
- Settings: smooth/medium/rough GRF, low/multi-frequency Fourier, Gaussian
  blobs, piecewise blocks, threshold/level-set, dipole/vortex-pair, and
  front/ring/shock.
- Regimes: low, medium, and high.

This gives 7 × 4 × 10 = 280 macro cases and 840 regime nodes. The full design
contains 2,000 samples per macro case, or 560,000 samples total.

Canonical arrays are channels-last:

```text
condition   [N, H, W, V_cond]
trajectory  [N, T, H, W, V_state]
geometry    [N, H, W, 1]
```

Static PDEs use `T=1`. Temporal PDEs use `T=9`; Navier–Stokes is never reduced
to an initial/final pair. Generated metadata explicitly records `T`. Evaluation
metadata records `mask_id`, deterministic `mask_seed`, observation count, and
observation ratio.

| Tier | Samples/macro case | Total samples | Intended use |
| --- | ---: | ---: | --- |
| tiny | 5 | 1,400 | unit/integration preflight |
| debug | 20 | 5,600 | local debugging |
| signal | 100 | 28,000 | first real experiments |
| medium | 500 | 140,000 | standard research |
| full | 2,000 | 560,000 | official leaderboard after validation |

## One-line interface

These commands are implemented without requiring YAML; `--config` remains the
advanced extension interface.

```bash
pdeobs download --tier medium --root ./data
pdeobs generate --tier signal --root ./data --num-workers 64

pdeobs generate-case \
  --pde navier_stokes --boundary periodic --setting vortex_pair \
  --param-regime high --num-samples 100 --root ./data

pdeobs train \
  --task sparse_recovery --model fno --data ./data/pdeobs_medium \
  --split iid --mask random_3pct --output runs/fno_sparse_recovery

pdeobs infer \
  --task sparse_recovery --model fno \
  --ckpt runs/fno_sparse_recovery/checkpoints/best.pt \
  --data ./data/pdeobs_medium --split test

pdeobs eval \
  --task sparse_recovery --pred runs/fno_sparse_recovery/preds.h5 \
  --data ./data/pdeobs_medium --metrics rel_l2,spectral,pde_residual

pdeobs benchmark \
  --preset fno_sparse_recovery --tier medium \
  --output runs/fno_sparse_recovery_benchmark
```

The download command has a stable default manifest endpoint, but there is no
official data release yet. It therefore returns a clear publication-gate error
until validated archives and checksums exist. A validated private manifest can
be passed with `--manifest`.

Prediction HDF5 files contain predictions, targets, sparse observations, masks,
geometry, sample IDs, and metadata. `eval --pred` streams these files, writes an
aggregate report, and writes per-sample JSONL suitable for `pdeobs analyze`.
A PDE residual is never inferred from prediction/target arrays alone: it is
reported unavailable unless a validated family-specific discrete operator and
the required physical context are present.

## Tasks and implementation status

| Task | Benchmark role | Current status |
| --- | --- | --- |
| Sparse-to-full recovery | primary field task | end-to-end runner, configs, anchors |
| Forward prediction | primary field task | end-to-end runner and FNO config |
| Inverse prediction | primary field task | end-to-end runner and Darcy/U-Net config |
| Semantic retrieval | lightweight anchor protocol | random/symbolic/ANN/VQ/RQ APIs and metrics; not a new method claim |
| Time-dependent world modeling | primary temporal task | rollout runner, short-train/long-test horizons, FNO/ConvLSTM/persistence APIs |
| Solver routing | lightweight anchor protocol | oracle/router/regret APIs; requires aligned solver-loss records |
| Foundation transfer | lightweight protocol | MAE-small and supervised encoder components; no foundation-model claim |

The field Trainer intentionally rejects retrieval/routing/transfer names. This
prevents an API helper from being misrepresented as a completed leaderboard.
Future methods can register entry points or provide explicit YAML without
changing the one-line core.

## Official splits

The seven official views are IID, boundary OOD, setting OOD, parameter OOD,
combination OOD, mask OOD, and time-horizon OOD. Generated samples carry the
factor membership flags. The runner evaluates paired IID/OOD rows and reports
oriented degradation.

Checked-in factor configs train separately for boundary, setting, parameter,
and combination OOD, so a held-out factor cannot leak through a different OOD
experiment. The mask configuration trains on 3% random observations and tests
1%, 5%, 10%, regular-grid, missing-block, line, boundary, and clustered masks.
Temporal configs train on horizons 1/2 and test 4/8.

Navier–Stokes low/medium/high currently changes both viscosity and initial-state
scale. It is therefore a **regime shift**, not an isolated single-parameter
causal claim.

## Anchor matrix

The minimum paper suite is zero fill, nearest, RBF, U-Net, FNO, CNO, continuous
ANN, flat VQ, RQ without prefix supervision, persistence, autoregressive U-Net,
autoregressive FNO, ConvLSTM, scratch, MAE-small, and supervised-multitask-small.

`configs/experiment/benchmark_paper_anchors.yaml` is the executable field-task
matrix. It includes transparent recovery anchors, learned recovery/forward/
inverse/rollout anchors, and separate factor/mask OOD runs. Retrieval, routing,
and transfer components are versioned APIs/protocols, not fabricated result
rows. Paper tables must be generated from checked-in configs and versioned run
artifacts; no completed leaderboard is currently claimed.

## Required difficulty analysis

The manuscript must report all 15 analyses:

1. observation-ratio difficulty (1%, 3%, 5%, 10%);
2. observation-pattern difficulty at comparable ratios;
3. PDE-family difficulty;
4. boundary generalization;
5. setting generalization;
6. physical-regime extrapolation;
7. combination OOD;
8. 1/2/4/8-step horizon difficulty;
9. low/mid/high-frequency error;
10. semantic ambiguity under sparse observation;
11. solver-routing accuracy and regret;
12. PDE × setting difficulty heatmaps by boundary;
13. data scaling (28k/140k/560k);
14. model scaling; and
15. qualitative failure cases with observation, target, prediction, error, and
    metadata.

`pdeobs analyze` provides deterministic factor, spectral, horizon, OOD, scaling,
and failure-ranking tables. Prediction evaluation now emits real per-sample
records for failure selection. Semantic ambiguity and routing tables require
their dedicated anchor outputs; they must remain empty rather than be invented
when those outputs are absent.

## Scientific publication gate

The 7 × 4 × 10 design and deterministic workflow are implemented. The bundled
compact solvers are still development references, not paper-grade ground truth.
Non-finite solver results now fail immediately instead of being replaced or
clipped, but that safety check is not numerical validation.

Before any official dataset or benchmark-paper result is published, complete
the convergence, residual, trusted-solver, conservation, and full-factor checks
in [NUMERICAL_VALIDATION.md](NUMERICAL_VALIDATION.md); freeze the exact commit
and environment; publish a versioned manifest with sizes and SHA-256 checksums;
and archive the result artifacts. Until then, the defensible claim is an
**executable controlled benchmark design/reference workflow**, not a released
560,000-sample ground-truth dataset.

## Claims that are out of scope

Do not claim the best semantic-ID model, the best world model, a new foundation
model, state of the art on every task, or convergence-validated ground truth
before the publication gate passes. The benchmark contribution stands on its
controlled factorization, reproducible tooling, official evaluation views, and
transparent difficulty analysis.
