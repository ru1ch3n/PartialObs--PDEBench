# PDE-OBS / PartialObs-PDEBench

**A controlled benchmark, reference implementation, and research map for PDE
learning under partial observation.**

[Project website](https://ru1ch3n.github.io/PartialObs--PDEBench/) |
[Benchmark Builder](https://ru1ch3n.github.io/PartialObs--PDEBench/builder/) |
[benchmark-paper contract](docs/BENCHMARK_PAPER.md) |
[observation-training protocol](docs/OBSERVATION_TRAINING_PROTOCOL.md) |
[reference protocol](docs/PROTOCOL.md) |
[numerical solver gate](docs/NUMERICAL_VALIDATION.md) |
[extension guide](docs/EXTENDING.md) |
[Linux server guide](docs/SERVER.md) |
[SeaWulf guide](hpc/seawulf/README.md)

PDE-OBS provides executable tools for factorized data generation, observation
masks, IID/OOD splits, recovery/forward/inverse and rollout baselines, metrics, and
cluster-scale experiments. The existing `docs/` research map remains part of
this repository.

> **Paper scope:** this repository treats
> **“PDE-OBS: A Controlled Partial-Observation Benchmark for PDE Dynamics”**
> as the only manuscript in scope. Its contribution is the dataset design,
> task/split/metric protocol, anchor leaderboard, difficulty analysis, and
> one-line tooling. Semantic-ID, large world-model, and foundation-model method
> claims are explicitly separate projects. Run `pdeobs protocol --check` to
> detect drift between the frozen paper contract and the default data config.

> **Scientific status:** the generator is undergoing numerical revalidation.
> Periodic solvers now follow the FNO/DiffusionPDE spectral protocols and
> bounded solvers use boundary-consistent finite-difference/finite-volume and
> topology-matched vorticity--streamfunction operators. They are **not paper
> ground truth until the
> checked-in convergence gate passes**. The active campaign is data-only: first
> the seven-PDE demo, then 20 samples for every factor combination (5,600 total).
> No model training and no 560,000-sample full generation should be submitted
> before that report is reviewed. See the
> [numerical solver gate](docs/NUMERICAL_VALIDATION.md).

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
- One CLI for protocol checks, planning, generation, one-case generation,
  download, training, inference, evaluation,
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
pdeobs generate --tier tiny --root ./data --num-workers 4
```

Generation is deterministic and resumable. Existing shards are skipped only
after their completion record and checksum pass validation.

For the current numerical-validation phase, use the dedicated configs instead
of the paper tiers:

```bash
# Seven PDEs, periodic demo.
pdeobs generate \
  --config configs/dataset/numerics_demo.yaml \
  --output datasets/numerics-demo

# Complete 280-factor coverage, 20 samples per macro case, 5,600 total.
# Use the SeaWulf submission workflow rather than running this on a laptop.
pdeobs plan \
  --config configs/dataset/numerics_validation20.yaml \
  --output datasets/plans/numerics-validation20.jsonl
```

The exact scratch-only SeaWulf commands, CPU policy, dependency windows, and
quality-report paths are in the [SeaWulf guide](hpc/seawulf/README.md).

For a tailored PDE/boundary/setting slice, use the
[interactive Benchmark Builder](https://ru1ch3n.github.io/PartialObs--PDEBench/builder/).
It generates local, Linux-server, and SeaWulf commands together with report and
strict quality profiles plus per-PDE loss reporting. The publication-candidate
choice is shown but intentionally blocked until a trusted external evidence
verifier and frozen per-stratum thresholds exist. Rejected samples are logged
in `*.quality-failures.jsonl`.

The same planner can describe `medium` and `full`, but do not publish those
outputs from the bundled compact solvers as benchmark ground truth. First
complete [the numerical validation gate](docs/NUMERICAL_VALIDATION.md). A
validated solver plugin is also required and must provide a registered
case-specific residual/solver-evidence contract.

The completed 5,600-sample SeaWulf factor gate and all seven normalized PDE
losses are recorded in the
[validation20 report](release/NUMERICS_VALIDATION20.md). Ground truth is always
solved and checksummed before any observation mask is applied; case-specific
solver routing never depends on the observation protocol.

For the primary IID observation comparison, normal trainable operator
baselines use a separate checkpoint for every PDE family and every one of the
nine observation masks; the training and evaluation masks match. The same
physical HDF5 dataset is reused for all masks--the workflow does not generate
nine copies of the fields. The existing random-3%-to-other-masks experiment is
retained as a separate mask-transfer/OOD analysis. See the
[observation-training protocol](docs/OBSERVATION_TRAINING_PROTOCOL.md) for the
method-specific exceptions, corrected job counts, and compute assumptions.

The requested ten-row campaign is partly a planning target. RBF and the compact
U-Net/FNO/CNO references are available now. Gappy POD, DeepONet, PINN/PINO,
Transolver/GNOT, DiffusionPDE, and FunDPS require reviewed external adapters;
the Builder reports those blockers and never emits fake training commands.

## One-line benchmark tools

The paper-facing interface does not require YAML. Advanced runs can still pass
`--config` and repeatable `--set` overrides.

```bash
# Publication-gated download interface. This becomes live when a validated
# release manifest and tier archives are published.
pdeobs download --tier medium --root ./data

# Canonical factorized generation. Output: ./data/pdeobs_signal
pdeobs generate --tier signal --root ./data --num-workers 64

# One explicit regime case. Output: ./data/pdeobs_cases/...
pdeobs generate-case \
  --pde navier_stokes \
  --boundary periodic \
  --setting vortex_pair \
  --param-regime high \
  --num-samples 100 \
  --root ./data

# Wheel-safe FNO reference preset. The preset chooses one shape-compatible
# Poisson reference case; use YAML for a full cross-factor experiment matrix.
pdeobs train \
  --task sparse_recovery \
  --model fno \
  --data ./data/pdeobs_medium \
  --split iid \
  --mask random_3pct \
  --output runs/fno_sparse_recovery

pdeobs infer \
  --task sparse_recovery \
  --model fno \
  --ckpt runs/fno_sparse_recovery/checkpoints/best.pt \
  --data ./data/pdeobs_medium \
  --split test

pdeobs eval \
  --task sparse_recovery \
  --pred runs/fno_sparse_recovery/preds.h5 \
  --data ./data/pdeobs_medium \
  --metrics rel_l2,spectral,pde_residual

pdeobs benchmark \
  --preset fno_sparse_recovery \
  --tier medium \
  --output runs/fno_sparse_recovery_benchmark
```

`eval --pred` writes both an aggregate JSON report and factor-aware per-sample
JSONL records for difficulty/failure analysis. `pde_residual` is reported as
unavailable until the prediction artifact carries a validated family-specific
discrete operator and all required physical context; the CLI never fabricates a
residual score.

## Train and evaluate a recovery baseline

```bash
# Generate the focused 34-sample signal-tier case. Unlike the five-sample
# tiny tier, this seed contains train, validation, and test rows for low regime.
pdeobs generate --config configs/dataset/recovery_signal.yaml \
  --output datasets/signal

pdeobs train --config configs/experiment/recovery_unet.yaml \
  --set data.root=datasets/signal \
  --set training.epochs=2 \
  --output runs/recovery-signal

pdeobs eval --config configs/experiment/recovery_unet.yaml \
  --set data.root=datasets/signal \
  --checkpoint runs/recovery-signal/checkpoints/best.pt \
  --output runs/recovery-signal/metrics.json

pdeobs infer --config configs/experiment/recovery_unet.yaml \
  --set data.root=datasets/signal \
  --checkpoint runs/recovery-signal/checkpoints/best.pt \
  --output runs/recovery-signal/predictions.h5
```

Prediction files include prediction, target, sparse observation, mask, geometry,
sample ID, and metadata. They are appended batch by batch and atomically
published as HDF5, so full-tier inference does not retain the complete result in
memory.

The default heat-rollout experiment uses the same strict signal-tier policy:

```bash
pdeobs generate --config configs/dataset/rollout_signal.yaml \
  --output datasets/signal
pdeobs train --config configs/experiment/rollout_fno.yaml
```

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
pdeobs protocol     print/check the frozen benchmark-paper contract
pdeobs doctor       verify a local or SeaWulf environment
pdeobs list         discover registered components
pdeobs plan         write explicit generation jobs for a Slurm array
pdeobs generate     produce one job, a plan, or a local tier
pdeobs generate-case generate one explicit factor/regime case
pdeobs download     fetch a checksum-verified published tier
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

The checked-in paper matrix is
`configs/experiment/benchmark_paper_anchors.yaml`. It includes transparent and
neural field anchors plus separately trained boundary-, setting-, parameter-,
combination-, and mask-OOD configurations. Retrieval, routing, and transfer are
lightweight protocol/API anchors in this benchmark paper; they are not presented
as completed new methods or a finished leaderboard.

## Observation-conditioned campaign planning

The proposed primary recovery table contains 10 method slots x 7 PDE families
x 9 observation masks = 630 IID result cells, after one implementation is
frozen for each ambiguous slot such as PINN versus PINO. Six normal trainable
slots require 378 neural fits at one seed, or 504 when only the PINN/PINO slot
uses three seeds. Gappy POD adds seven leakage-free training-split fits.
DiffusionPDE and FunDPS add zero prior-training jobs when compatible upstream
checkpoints exist, or up to fourteen per-PDE prior jobs when both must be
trained. Total preparation is therefore 385-399 jobs at one seed or 511-525
with the special three-seed PINN/PINO rule.

The recommended ten-day planning scenario is the 140,000-record `medium` tier
for the complete ten-slot/nine-mask comparison, followed by a reduced
560,000-record full-tier anchor table. Medium contains 20,000 records per PDE
and approximately 14,000 optimizer-training records per PDE; full contains
80,000 and exactly 56,000 respectively. Runtime figures in the protocol are
unmeasured A6000 capacity estimates, not SeaWulf/A100 guarantees. Every real
campaign must begin with a measured pilot, equal-seed uncertainty reporting,
and the dataset-quality gate.

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

## Run on a Linux server

On an ordinary Linux CPU/GPU server, keep code, data, and runs separate and use
`tmux` so an SSH disconnect does not stop a job:

```bash
git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git
cd PartialObs--PDEBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[train,test]"

export PDEOBS_DATA="$PWD/datasets"
export PDEOBS_RUNS="$PWD/runs"
pdeobs doctor                         # add --gpu when CUDA is expected
```

Start a persistent terminal with `tmux new -s pdeobs`, then run this inside
the new session:

```bash
source .venv/bin/activate
export PDEOBS_DATA="$PWD/datasets"
export PDEOBS_RUNS="$PWD/runs"
pdeobs generate --config configs/dataset/smoke.yaml \
  --output "$PDEOBS_DATA/smoke"
pdeobs aggregate --input "$PDEOBS_DATA/smoke" \
  --output "$PDEOBS_DATA/smoke/summary.json" --validate-shards
pdeobs train --config configs/experiment/recovery_unet_smoke.yaml \
  --output "$PDEOBS_RUNS/smoke-train"
```

Detach from `tmux` with `Ctrl-b d` and reconnect with
`tmux attach -t pdeobs`. For the strict signal-tier workflow, GPU checks,
evaluation, resume commands, and update procedure, use the complete
[Linux server guide](docs/SERVER.md).

## SeaWulf quick start

SeaWulf work must be submitted through Slurm. The repository includes CPU
generation/aggregation launchers and A100 training/evaluation launchers. After
cloning the repository on `milan.seawulf.stonybrook.edu`, set your group name
and run:

```bash
module load slurm
cd PartialObs--PDEBench

export PDEOBS_GROUP=YOUR_GROUP
export PDEOBS_COMMIT="$(git rev-parse --short=12 HEAD)"
export PDEOBS_ENV="/gpfs/projects/$PDEOBS_GROUP/envs/pdeobs-$PDEOBS_COMMIT"
export PDEOBS_DATA="/gpfs/scratch/$USER/pdeobs/data"
export PDEOBS_RUNS="/gpfs/scratch/$USER/pdeobs/runs"
mkdir -p logs "$(dirname "$PDEOBS_ENV")" \
  "$PDEOBS_DATA/plans" "$PDEOBS_RUNS"

# Bootstrap only inside a compute allocation, never on the login node.
srun --partition=short-40core-shared --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash -l
bash hpc/seawulf/bootstrap.sh
exit

# Make an exact smoke plan, then chain generation, validation, and training.
"$PDEOBS_ENV/bin/python" -m pdeobs plan \
  --config configs/dataset/smoke.yaml --tier tiny \
  --output "$PDEOBS_DATA/plans/smoke.jsonl"
smoke_job="$(sbatch --parsable --array=0-0 \
  hpc/seawulf/generate_array.sbatch \
  configs/dataset/smoke.yaml "$PDEOBS_DATA/smoke" \
  "$PDEOBS_DATA/plans/smoke.jsonl")"
smoke_job="${smoke_job%%;*}"
check_job="$(sbatch --parsable --dependency="afterok:$smoke_job" \
  hpc/seawulf/aggregate_cpu.sbatch \
  "$PDEOBS_DATA/smoke" "$PDEOBS_DATA/smoke/summary.json" \
  "$PDEOBS_DATA/plans/smoke.jsonl")"
check_job="${check_job%%;*}"
train_job="$(sbatch --parsable --dependency="afterok:$check_job" \
  hpc/seawulf/train_gpu.sbatch \
  configs/experiment/recovery_unet_smoke.yaml \
  --output "$PDEOBS_RUNS/smoke-train")"
train_job="${train_job%%;*}"
echo "generation=$smoke_job validation=$check_job training=$train_job"
squeue --user="$USER"
```

Do not submit the full dataset first. Inspect `logs/`, the validation summary,
and `seff JOB_ID`; then follow the exact-commit, signal-tier, array-window,
resume, storage, and archive instructions in the full
[SeaWulf guide](hpc/seawulf/README.md). SeaWulf scratch is temporary and not
backed up.

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

No convergence-validated dataset release is published yet. Consequently,
the manifest-free `pdeobs download --tier ...` interface targets a stable,
publication-gated release URL and currently fails with an explicit validation
message. A private validated release can be used with `--manifest URL_OR_PATH`.
This interface will become live without a CLI change once checksummed tier
artifacts and the validation report are published.

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
