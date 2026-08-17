# Running PDE-OBS on SeaWulf

These launchers target Stony Brook's current SeaWulf Slurm environment. Verify
the queue and module names before a large campaign because cluster policy can
change.

## Current phase: dataset generation only

The current campaign does **not** submit model training. First validate the
revised numerical solvers with 20 samples for every PDE x boundary x setting
combination. Do not submit the 560,000-sample paper tier until its PDE-loss and
refinement reports are accepted. See
[`docs/NUMERICAL_VALIDATION.md`](../../docs/NUMERICAL_VALIDATION.md).

For a scratch-only working layout (checkout, environment, caches, data, runs,
and logs), use:

```bash
export PDEOBS_BASE="/gpfs/scratch/$USER/pdeobs"
export PDEOBS_REPO="$PDEOBS_BASE/PartialObs--PDEBench"
export PDEOBS_DATA="$PDEOBS_BASE/data"
export PDEOBS_RUNS="$PDEOBS_BASE/runs"
export PDEOBS_COMMIT="$(git rev-parse --short=12 HEAD)"
export PDEOBS_ENV="$PDEOBS_BASE/envs/pdeobs-$PDEOBS_COMMIT"
export CONDA_PKGS_DIRS="$PDEOBS_BASE/conda-pkgs"
export PIP_CACHE_DIR="$PDEOBS_BASE/pip-cache"
export XDG_CACHE_HOME="$PDEOBS_BASE/cache"
mkdir -p "$PDEOBS_DATA/plans" "$PDEOBS_RUNS" logs \
  "$(dirname "$PDEOBS_ENV")" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
```

Scratch is not backed up and is purged under current site policy. Push code to
GitHub before submission and copy accepted reports/data to independent archival
storage before the purge window.

### A. Seven-PDE numerical demo

Use the resolved configuration written beside the plan; using
`configs/dataset/default.yaml` after a tier override would fail provenance
matching.

```bash
"$PDEOBS_ENV/bin/python" -m pdeobs plan \
  --config configs/dataset/numerics_demo.yaml \
  --output "$PDEOBS_DATA/plans/numerics-demo.jsonl"

demo_job="$(sbatch --parsable --cpus-per-task=1 --mem=4G \
  --array=0-20%7 hpc/seawulf/generate_array.sbatch \
  "$PDEOBS_DATA/plans/numerics-demo.resolved.yaml" \
  "$PDEOBS_DATA/numerics-demo" \
  "$PDEOBS_DATA/plans/numerics-demo.jsonl")"
demo_job="${demo_job%%;*}"

demo_check="$(sbatch --parsable --cpus-per-task=1 --mem=8G \
  --dependency="afterok:$demo_job" hpc/seawulf/aggregate_cpu.sbatch \
  "$PDEOBS_DATA/numerics-demo" \
  "$PDEOBS_DATA/numerics-demo/summary.json" \
  "$PDEOBS_DATA/plans/numerics-demo.jsonl" \
  --quality-strict --require-all-pdes)"
demo_check="${demo_check%%;*}"
squeue -j "$demo_job,$demo_check"
```

Inspect `summary.json`, `summary.quality.json`, `summary.quality.csv`, every
`*.quality-failures.jsonl`, Slurm logs, `sacct`, and `seff`. A successful job
state alone is not scientific acceptance.

### B. Complete 20-sample factor validation

Only after the demo passes:

```bash
export PDEOBS_GENERATION_CONCURRENCY=20
bash hpc/seawulf/submit_validation20.sh
```

The script verifies exactly 840 generation rows and 5,600 samples, submits nine
dependency-chained windows of at most 100 single-core tasks, then submits one
strict aggregate job. It writes
`$PDEOBS_DATA/numerics-validation20.campaign.txt`. No training job is submitted.
Use `PDEOBS_GENERATION_CONCURRENCY=8` for the first filesystem pilot if the
current queue or GPFS load is high; increase only after `seff`/throughput data.

### C. Approval boundary

There is intentionally no automatic command on this page that converts the
validation run into the true full campaign. Review the per-PDE and per-stratum
losses, convergence trends, divergence, BC errors, failures, and checksums
first. A later approved full plan must keep a separate output directory and
must pin the accepted commit and threshold/evidence table.

## 1. Clone an exact Git revision

From your local computer, connect to the Milan login node. Then clone and pin
the campaign revision:

```bash
ssh YOUR_NETID@milan.seawulf.stonybrook.edu
module load slurm
git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git
cd PartialObs--PDEBench
git checkout YOUR_RELEASE_TAG_OR_COMMIT
```

Queued jobs do not pull from Git. This ensures every shard in a campaign uses
the revision recorded at submission.

## 2. Choose persistent and scratch locations

```bash
export PDEOBS_GROUP=YOUR_GROUP
export PDEOBS_COMMIT="$(git rev-parse --short=12 HEAD)"
export PDEOBS_ENV="/gpfs/projects/$PDEOBS_GROUP/envs/pdeobs-$PDEOBS_COMMIT"
export PDEOBS_DATA="/gpfs/scratch/$USER/pdeobs/data"
export PDEOBS_RUNS="/gpfs/scratch/$USER/pdeobs/runs"
mkdir -p logs "$(dirname "$PDEOBS_ENV")" \
  "$PDEOBS_DATA/plans" "$PDEOBS_RUNS"
```

Run these exports again after each fresh login (or source them from a private,
untracked shell snippet). Do not commit account or group-specific paths.

SeaWulf scratch is temporary, is not backed up, and is subject to a 45-day
purge. Project space lasts for the project and is appropriate for a shared
environment or curated copy, but it is not backed up either. Keep at least one
verified copy of irreplaceable results in an independent off-cluster archive.

## 3. Build once in a compute allocation

Do not build packages on a login node. From the repository root, request an
interactive CPU allocation, bootstrap, and return to the login node:

```bash
module load slurm
srun --partition=short-40core-shared --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash -l
bash hpc/seawulf/bootstrap.sh
"$PDEOBS_ENV/bin/python" -m pdeobs doctor --cluster seawulf --offline
"$PDEOBS_ENV/bin/python" -m pdeobs protocol --check
exit
```

The bootstrap builds a wheel from the checked-out commit and installs it
non-editably. It records that commit inside `$PDEOBS_ENV`; every job refuses to
run when the environment and checkout revisions differ. Use a commit-specific
environment path as above, or rerun the bootstrap after changing revisions.

`environment.yml` selects the reference PyTorch 2.5 / CUDA 12.4 environment. Set
`PDEOBS_CUDA_MODULE` if the selected SeaWulf architecture requires an explicit
CUDA module, and confirm the available module/driver combination before setup.
For a GPU preflight, set `PDEOBS_REQUIRE_GPU=1` inside a GPU allocation; the GPU
job scripts also stop immediately if PyTorch cannot see CUDA.

If compute nodes cannot access package servers, populate a compatible
wheelhouse, including `setuptools` and `wheel`, and set `PDEOBS_WHEELHOUSE`
before running the bootstrap script. For an archival campaign, also save an
exact environment export beside its plan and validation report.

## 4. Run the dependency-chained smoke workflow

```bash
"$PDEOBS_ENV/bin/python" -m pdeobs plan \
  --config configs/dataset/smoke.yaml --tier tiny \
  --output "$PDEOBS_DATA/plans/smoke.jsonl"

generation_job="$(sbatch --parsable --array=0-0 \
  hpc/seawulf/generate_array.sbatch \
  configs/dataset/smoke.yaml "$PDEOBS_DATA/smoke" \
  "$PDEOBS_DATA/plans/smoke.jsonl")"
generation_job="${generation_job%%;*}"

aggregation_job="$(sbatch --parsable \
  --dependency="afterok:${generation_job}" \
  hpc/seawulf/aggregate_cpu.sbatch \
  "$PDEOBS_DATA/smoke" "$PDEOBS_DATA/smoke/summary.json" \
  "$PDEOBS_DATA/plans/smoke.jsonl")"
aggregation_job="${aggregation_job%%;*}"

training_job="$(sbatch --parsable \
  --dependency="afterok:${aggregation_job}" \
  hpc/seawulf/train_gpu.sbatch \
  configs/experiment/recovery_unet_smoke.yaml \
  --output "$PDEOBS_RUNS/smoke-train")"
training_job="${training_job%%;*}"

echo "generation=$generation_job aggregation=$aggregation_job training=$training_job"
squeue -j "$generation_job,$aggregation_job,$training_job"
```

The aggregate job validates checksums and exact plan coverage. Because each
step uses `afterok`, a failure prevents downstream work from starting. Inspect
`logs/`, `sacct -j JOB_ID`, and `seff JOB_ID` before proceeding.

## 5. Run the strict signal-tier recovery example

This focused example generates one 34-sample low-regime shard with real
train/validation/test coverage, validates it, trains, and evaluates:

```bash
"$PDEOBS_ENV/bin/python" -m pdeobs plan \
  --config configs/dataset/recovery_signal.yaml --tier signal \
  --output "$PDEOBS_DATA/plans/recovery-signal.jsonl"

signal_job="$(sbatch --parsable --array=0-0 \
  hpc/seawulf/generate_array.sbatch \
  configs/dataset/recovery_signal.yaml "$PDEOBS_DATA/signal" \
  "$PDEOBS_DATA/plans/recovery-signal.jsonl")"
signal_job="${signal_job%%;*}"

signal_check_job="$(sbatch --parsable --dependency="afterok:${signal_job}" \
  hpc/seawulf/aggregate_cpu.sbatch \
  "$PDEOBS_DATA/signal" "$PDEOBS_DATA/signal/summary.json" \
  "$PDEOBS_DATA/plans/recovery-signal.jsonl")"
signal_check_job="${signal_check_job%%;*}"

signal_train_job="$(sbatch --parsable --dependency="afterok:${signal_check_job}" \
  hpc/seawulf/train_gpu.sbatch configs/experiment/recovery_unet.yaml \
  --output "$PDEOBS_RUNS/recovery-signal")"
signal_train_job="${signal_train_job%%;*}"

signal_eval_job="$(sbatch --parsable --dependency="afterok:${signal_train_job}" \
  hpc/seawulf/evaluate_gpu.sbatch configs/experiment/recovery_unet.yaml \
  "$PDEOBS_RUNS/recovery-signal/checkpoints/best.pt" \
  --output "$PDEOBS_RUNS/recovery-signal/metrics.json")"
signal_eval_job="${signal_eval_job%%;*}"

echo "generation=$signal_job validation=$signal_check_job training=$signal_train_job evaluation=$signal_eval_job"
squeue -j "$signal_job,$signal_check_job,$signal_train_job,$signal_eval_job"
```

## 6. Scale to a factorized tier

```bash
"$PDEOBS_ENV/bin/python" -m pdeobs plan \
  --config configs/dataset/default.yaml --tier tiny \
  --output "$PDEOBS_DATA/plans/tiny.jsonl"

bash hpc/seawulf/submit_generation.sh \
  "$PDEOBS_DATA/plans/tiny.jsonl" \
  configs/dataset/default.yaml \
  "$PDEOBS_DATA/tiny" \
  0 99
```

Each array element writes an independent shard. Array tasks never append to a
shared HDF5 or CSV file. A per-shard ownership lock rejects accidentally
overlapping submissions.

The plan produced from `configs/dataset/default.yaml` has 840 independent
regime-node jobs. As a conservative safety cap, `submit_generation.sh` accepts
an inclusive `START STOP` window and refuses windows larger than
`PDEOBS_MAX_QUEUED_TASKS` (default 100).
Wait for `0 99` to finish, then submit `100 199`, and continue through the last
plan index; `%${PDEOBS_GENERATION_CONCURRENCY:-4}` in the array request limits
simultaneously running tasks. Confirm current site policy before changing either
limit.

The 560,000-sample full tier is approximately **296 GiB raw** before HDF5
compression. Compression depends strongly on the generated fields. Budget
additional scratch space for partial shards, logs, metadata, and any curated
copy to project storage; generate `tiny` first and measure its realized size
before reserving space for `full`.

After every array window succeeds, strictly validate the exact plan:

```bash
aggregation_job="$(sbatch --parsable hpc/seawulf/aggregate_cpu.sbatch \
  "$PDEOBS_DATA/tiny" \
  "$PDEOBS_DATA/tiny/summary.json" \
  "$PDEOBS_DATA/plans/tiny.jsonl")"
```

The aggregate job verifies completion records, checksums, sample identities,
and exact plan coverage. It is a storage/provenance gate, not a substitute for
the numerical validation required for paper data.

## 7. Plan the nine-observation training campaign

The primary IID comparison uses a separate checkpoint for every normal
trainable method, PDE family, and observation mask. It reuses one validated
physical dataset; masks are deterministic views and must not be materialized as
nine duplicate datasets. The random-3%-trained configuration remains a
secondary mask-transfer/OOD experiment.

The complete requested ten-slot suite is a planning target, not a submit-ready
SeaWulf array. This repository currently executes RBF and compact-reference
U-Net/FNO/CNO. Gappy POD, DeepONet, PINN/PINO, Transolver/GNOT, DiffusionPDE,
and FunDPS need reviewed adapters and frozen upstream versions/checkpoints.
Never submit planning-only rows or silently replace them with a different
method. Use `configs/campaign/core_observation_medium.yaml` as a manifest of
the intended policy, then create explicit executable experiment configs only
for integrated methods.

For a ten-day study, start with the 140,000-record medium tier, pilot one PDE
and three masks, and measure wall time, memory, I/O, and failure rate before
expanding. The attachment's capacity scenario assumes twelve dedicated A6000s;
SeaWulf launchers use the shared A100 partition, one GPU per job, and do not
guarantee twelve concurrent GPUs. Treat every A6000 GPU-hour range as an
unmeasured planning estimate and derive SeaWulf resources from the pilot.

Keep campaign outputs unique by method, PDE, mask, and seed. Submit bounded
windows, wait for each window to finish, and chain evaluation only after the
matching dataset checksum/quality gate and checkpoint succeed. See
`docs/OBSERVATION_TRAINING_PROTOCOL.md` for corrected result/job counts,
method-specific reuse rules, Navier-Stokes representation constraints, and the
reduced full-tier anchor recommendation.

## 8. Monitor, resume, and evaluate

```bash
squeue --user="$USER"
sacct -j JOB_ID
seff JOB_ID

sbatch hpc/seawulf/train_gpu.sbatch \
  configs/experiment/recovery_unet.yaml \
  --output "$PDEOBS_RUNS/recovery-signal" \
  --resume "$PDEOBS_RUNS/recovery-signal/checkpoints/last.pt"
```

The default A100 scripts request one GPU, eight CPU cores, 64 GB memory, and
eight hours. Change resources at submission time when measurements justify it;
variables are not expanded in `#SBATCH` headers. The trainer writes resumable
checkpoints; if a run reaches the time limit, resubmit with
`--resume /path/to/checkpoints/last.pt`. Always ensure a training configuration's
`data.root` matches the tier you generated; the checked-in recovery example
uses `$PDEOBS_DATA/signal`.

Official references:

- [SeaWulf usage guidance](https://rci.stonybrook.edu/HPC/about/guidance)
- [queues](https://rci.stonybrook.edu/HPC/docs/architecture/queues-table)
- [Slurm jobs](https://rci.stonybrook.edu/HPC/docs/jobs/slurm-overview)
- [storage and retention](https://rci.stonybrook.edu/HPC/docs/storage/layout)
- [Conda environments](https://rci.stonybrook.edu/HPC/docs/software/conda)
