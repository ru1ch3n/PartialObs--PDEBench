# PDE-OBS / PartialObs-PDEBench

**A controlled benchmark, reference implementation, and research map for PDE
learning under partial observation.**

[Project website](https://ru1ch3n.github.io/PartialObs--PDEBench/) ·
[reference protocol](docs/PROTOCOL.md) ·
[extension guide](docs/EXTENDING.md) ·
[SeaWulf guide](hpc/seawulf/README.md)

PDE-OBS provides executable tools for factorized data generation, observation
masks, IID/OOD splits, recovery/forward/inverse and rollout baselines, metrics, and
cluster-scale experiments. The existing `docs/` research map remains part of
this repository.

> **Scientific status:** version 0.1 contains compact, deterministic reference
> solvers and baselines. They are suitable for pipeline development, CI, and
> method prototyping. They are **not yet convergence-validated paper-data
> solvers**. A scientific release must replace or validate them against trusted
> high-resolution codes and publish the residual/convergence suite described in
> [the numerical validation gate](docs/NUMERICAL_VALIDATION.md).

## What is implemented

- Seven PDE families: Darcy, Poisson, Helmholtz, heat, reaction-diffusion,
  Burgers, and Navier-Stokes.
- Four boundary protocols, ten condition generators, and three physical
  regimes behind registries that can be extended without changing the CLI.
- Nested, deterministic generation plans from the 1,400-sample tiny tier
  through the intended 560,000-sample full design, plus a two-sample smoke case.
- Static fields and temporal trajectories in atomic, checksum-verified HDF5
  shards with metadata and deterministic semantic IDs.
- Random-ratio, exact-count, grid, missing-block, line, boundary, and clustered
  observation masks.
- IID, boundary-, setting-, parameter-, combination-, mask-, and horizon-OOD
  views.
- Transparent interpolation/persistence baselines and compact U-Net, FNO,
  CNO-like, ConvLSTM/autoregressive, residual-encoder, and MAE-style anchors.
- Recovery, physical, spectral, rollout, retrieval, OOD, and solver-routing
  metrics, including continuous ANN, flat-VQ, and unsupervised-RQ anchors.
- One CLI for planning, generation, download, training, inference, evaluation,
  aggregation, problem-difficulty analysis, diagnostics, and component discovery.
- CPU/GPU Slurm scripts for SeaWulf, including manifest-driven arrays and
  resume-safe independent shards.

## Install directly from Git

Python 3.10 or newer is required.

```bash
git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git
cd PartialObs--PDEBench
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,test]"
pdeobs doctor
```

Later, use the same checkout as the source of truth:

```bash
git pull --ff-only
python -m pip install -e ".[train,test]"
```

For a paper run, checkout a tag or commit instead of following a moving branch.

## Five-minute smoke run

```bash
# Generate a two-sample, 16x16 Poisson shard.
pdeobs generate \
  --config configs/dataset/smoke.yaml \
  --output datasets/smoke

# Validate the output and summarize its manifest.
pdeobs aggregate \
  --input datasets/smoke \
  --output datasets/smoke/summary.json \
  --validate-shards

# See every registered solver, mask, method, and metric.
pdeobs list
```

For a full factorized tiny tier, use the default configuration:

```bash
pdeobs plan --config configs/dataset/default.yaml --tier tiny \
  --output datasets/plans/tiny.jsonl
pdeobs generate --config configs/dataset/default.yaml \
  --tier tiny --output datasets/tiny
```

Generation is deterministic and resumable. Existing shards are skipped only
after their completion record and checksum pass validation.

The same planner can describe `medium` and `full`, but do not publish those
outputs from the bundled compact solvers as benchmark ground truth. First
complete [the numerical validation gate](docs/NUMERICAL_VALIDATION.md), or
install a validated solver plugin at the same registered family names.

## Train and evaluate a recovery baseline

```bash
pdeobs train --config configs/experiment/recovery_unet.yaml \
  --set data.root=datasets/tiny \
  --set training.epochs=2

pdeobs eval --config configs/experiment/recovery_unet.yaml \
  --checkpoint runs/<run-id>/checkpoints/best.pt

pdeobs infer --config configs/experiment/recovery_unet.yaml \
  --checkpoint runs/<run-id>/checkpoints/best.pt \
  --output predictions/recovery.h5
```

Prediction files are appended batch by batch and atomically published as HDF5,
so full-tier inference does not retain the complete result in memory.

Every run records the fully resolved configuration, Git revision, dependency
versions, seed, host, and Slurm identifiers. The compact neural models here are
benchmark anchors, not claims of exact reproduction of all upstream recipes.

## Data layout

Canonical shard arrays use channels-last storage:

```text
condition   [N, H, W, V_cond]
trajectory  [N, T, H, W, V_state]
geometry    [N, H, W, 1]
```

Static PDEs have `T=1`; temporal references retain multiple states. Metadata
include family, boundary, setting, regime, physical parameters, split views,
solver version, seed, and semantic ID. At 128 by 128, the primary training mask
uses exactly 500 spatial points. See [the frozen protocol](docs/PROTOCOL.md) for
all tier and OOD definitions.

## Command-line interface

```text
pdeobs doctor       verify a local or SeaWulf environment
pdeobs list         discover registered components
pdeobs plan         write explicit generation jobs for a Slurm array
pdeobs generate     produce one job, a plan, or a local tier
pdeobs download     fetch a published tier with SHA-256 verification
pdeobs train        fit a configured method with resumable checkpoints
pdeobs infer        write predictions from a checkpoint
pdeobs eval         calculate benchmark metrics and OOD breakdowns
pdeobs benchmark    run configured methods/splits as a local benchmark
pdeobs aggregate    strictly validate shards/plans and combine result records
pdeobs analyze      summarize factor difficulty, scaling, OOD, and failures
```

Run `pdeobs <command> --help` for all options. YAML supports environment values
such as `${PDEOBS_DATA}` and repeatable `--set key.path=value` overrides.
Problem-difficulty tables are documented in
[docs/DIFFICULTY_ANALYSIS.md](docs/DIFFICULTY_ANALYSIS.md).

## Extensibility

PDE solvers and benchmark methods use Python registries and standard package
entry points. A future method can live in this repository or in a separate
package and declare:

```toml
[project.entry-points."pdeobs.methods"]
my_method = "my_package.pdeobs_plugin:register"
```

No CLI edits are required. Read [docs/EXTENDING.md](docs/EXTENDING.md) for the
component contract and testing checklist.

## SeaWulf

The repository contains separate shared-CPU, A100, aggregation, and
manifest-array launchers. The intended flow is clone Git, checkout an exact
revision, build the environment once in persistent storage, generate into
scratch, and copy curated releases/results to project storage. Start with
[hpc/seawulf/README.md](hpc/seawulf/README.md); do not launch the full tier before
the checked-in smoke case succeeds.

## Repository structure

```text
src/pdeobs/          package, solvers, masks, methods, metrics, runners
configs/             dataset, method, experiment, and SeaWulf YAML
hpc/seawulf/         environment and Slurm launchers
tests/               deterministic unit and end-to-end smoke tests
docs/                benchmark protocol, extension guide, and research website
scripts/             research-map maintenance utilities
data/curations/      curated research-map records
```

Git contains code, configurations, small fixtures, manifests, and checksums.
Generated datasets, model weights, run directories, environments, and container
images are intentionally ignored.

## Research website

The static research map is served from `docs/`. To rebuild it:

```bash
python -m pip install -e ".[site]"
python scripts/validate_papers.py
python scripts/generate_research_site.py
python -m http.server 8000 --directory docs
```

## License and citation

Repository code is MIT licensed. External datasets, weights, and upstream code
retain their original licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Citation metadata are in [CITATION.cff](CITATION.cff).
