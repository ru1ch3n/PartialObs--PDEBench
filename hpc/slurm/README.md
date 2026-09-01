# Running PDE-OBS on a Slurm HPC system

The launchers in this directory are portable Slurm templates. They intentionally
do not name a cluster, partition, account, QOS, module, filesystem, or GPU
model. Site policy always wins: consult the local administrator's documentation,
run a small pilot, and pass site-specific values through environment variables
or normal `sbatch` options.

## 1. Define a site profile

Set persistent storage paths before entering an allocation. The paths below are
examples; replace them with locations approved by your HPC site.

```bash
export PDEOBS_ENV="${PROJECT:-$HOME}/pdeobs/env"
export PDEOBS_DATA="${SCRATCH:-$PWD}/pdeobs/data"
export PDEOBS_RUNS="${SCRATCH:-$PWD}/pdeobs/runs"

# Optional scheduler routing. Leave unset to use scheduler defaults.
export PDEOBS_CPU_PARTITION="YOUR_CPU_PARTITION"
export PDEOBS_GPU_PARTITION="YOUR_GPU_PARTITION"
export PDEOBS_ACCOUNT="YOUR_ACCOUNT"
# export PDEOBS_QOS="YOUR_QOS"
# export PDEOBS_RESERVATION="YOUR_RESERVATION"

# Optional software setup. These are examples, not required module names.
# export PDEOBS_MODULES="python cuda"
# export PDEOBS_MODULE_PURGE=1
# export PDEOBS_MODULE_SETUP="$HOME/pdeobs-site-modules.sh"
```

`PDEOBS_MODULE_SETUP`, when set, must point to a readable shell file maintained
by the user or site. This avoids embedding site module names in the repository.
If Slurm commands are not on `PATH`, set `PDEOBS_SBATCH`, `PDEOBS_SRUN`,
`PDEOBS_SQUEUE`, and `PDEOBS_SACCT` to their absolute paths.

## 2. Build an immutable environment

Clone the repository on a compute node or shared filesystem, check out the
desired release commit, and start a short Slurm allocation according to local
policy. From inside that allocation:

```bash
git status --short
export PDEOBS_ENV_FILE=environment-generation.yml
bash hpc/slurm/bootstrap.sh
"$PDEOBS_ENV/bin/python" -m pdeobs doctor --cluster slurm --offline
```

Use `environment.yml` instead when learned baselines and CUDA are required. Set
`PDEOBS_REQUIRE_GPU=1` before bootstrap to require a working PyTorch GPU. The
bootstrap script builds a wheel, installs it outside the checkout, records the
exact Git commit, and refuses a dirty source tree.

## 3. Run a dependency-gated smoke test

Create the exact generation manifest first:

```bash
mkdir -p "$PDEOBS_DATA/plans" "$PDEOBS_DATA/smoke" logs
"$PDEOBS_ENV/bin/python" -m pdeobs plan \
  --config configs/dataset/smoke.yaml \
  --output "$PDEOBS_DATA/plans/smoke.jsonl"
```

Build scheduler-routing arguments once. Empty variables are omitted, so the
scheduler default remains valid:

```bash
site_args=()
[[ -n "${PDEOBS_CPU_PARTITION:-}" ]] && site_args+=(--partition="$PDEOBS_CPU_PARTITION")
[[ -n "${PDEOBS_ACCOUNT:-}" ]] && site_args+=(--account="$PDEOBS_ACCOUNT")
[[ -n "${PDEOBS_QOS:-}" ]] && site_args+=(--qos="$PDEOBS_QOS")
[[ -n "${PDEOBS_RESERVATION:-}" ]] && site_args+=(--reservation="$PDEOBS_RESERVATION")

generation_job="$(sbatch --parsable "${site_args[@]}" \
  --array=0-20%4 \
  hpc/slurm/generate_array.sbatch \
  configs/dataset/smoke.yaml \
  "$PDEOBS_DATA/smoke" \
  "$PDEOBS_DATA/plans/smoke.jsonl")"
generation_job="${generation_job%%;*}"

quality_job="$(sbatch --parsable "${site_args[@]}" \
  --dependency="afterok:${generation_job}" \
  hpc/slurm/aggregate_cpu.sbatch \
  "$PDEOBS_DATA/smoke" \
  "$PDEOBS_DATA/smoke/summary.json" \
  "$PDEOBS_DATA/plans/smoke.jsonl" \
  --quality-strict --require-all-pdes)"
quality_job="${quality_job%%;*}"

squeue -j "$generation_job,$quality_job"
```

The `afterok` dependency prevents aggregation from running after a failed array.
The generation workflow does **not** submit model training.

## 4. Submit bounded generation windows

Many sites cap queued array elements. `submit_generation.sh` therefore submits
one explicit window and refuses a window larger than
`PDEOBS_MAX_QUEUED_TASKS`:

```bash
export PDEOBS_GENERATION_CONCURRENCY=4
export PDEOBS_MAX_QUEUED_TASKS=100
bash hpc/slurm/submit_generation.sh \
  "$PDEOBS_DATA/plans/smoke.jsonl" \
  configs/dataset/smoke.yaml \
  "$PDEOBS_DATA/smoke" \
  0 20
```

Do not submit the next window until scheduler accounting confirms every element
in the current window completed successfully.

## 5. Train and evaluate on a GPU

The templates request one generic Slurm GPU and do not constrain its model.
Override memory, time, CPU count, partition, account, or GPU constraints on the
`sbatch` command line after a measured pilot.

```bash
gpu_args=(--gpus=1 --cpus-per-task=8 --mem=64G --time=08:00:00)
[[ -n "${PDEOBS_GPU_PARTITION:-}" ]] && gpu_args+=(--partition="$PDEOBS_GPU_PARTITION")
[[ -n "${PDEOBS_ACCOUNT:-}" ]] && gpu_args+=(--account="$PDEOBS_ACCOUNT")
[[ -n "${PDEOBS_QOS:-}" ]] && gpu_args+=(--qos="$PDEOBS_QOS")

train_job="$(sbatch --parsable "${gpu_args[@]}" \
  hpc/slurm/train_gpu.sbatch configs/experiment/recovery_unet_smoke.yaml)"
train_job="${train_job%%;*}"

sbatch "${gpu_args[@]}" --dependency="afterok:${train_job}" \
  hpc/slurm/evaluate_gpu.sbatch \
  configs/experiment/recovery_unet_smoke.yaml \
  "$PDEOBS_RUNS/YOUR_RUN/checkpoint.pt"
```

The training and evaluation launchers fail if CUDA is unavailable. Add a site
constraint such as `--constraint=...` only when required by local policy.

## 6. Larger dataset-only campaigns

Two guarded helpers preserve the checked-in scientific contracts:

```bash
bash hpc/slurm/submit_validation20.sh
bash hpc/slurm/submit_full_t15.sh
bash hpc/slurm/monitor_full_t15.sh "$PDEOBS_DATA/YOUR_CAMPAIGN.campaign.txt"
```

Both helpers verify the expected plan cardinality, reject duplicate campaign
records, bound queued work, and gate aggregation on successful generation. They
remain dataset-only workflows. Tune these variables from a pilot and site limits:

- `PDEOBS_CPU_PARTITION`, `PDEOBS_ACCOUNT`, `PDEOBS_QOS`, `PDEOBS_RESERVATION`
- `PDEOBS_GENERATION_CONCURRENCY`, `PDEOBS_GENERATION_WINDOW_SIZE`
- `PDEOBS_MAX_QUEUED_TASKS`
- `PDEOBS_FULL_BUNDLE_SIZE`, `PDEOBS_FULL_CPUS_PER_TASK`
- `PDEOBS_FULL_CONCURRENCY`, `PDEOBS_FULL_MEMORY`, `PDEOBS_FULL_TIME_LIMIT`

## Operational rules

- Never run long work on a login node.
- Keep datasets and checkpoints outside Git.
- Treat scratch storage as non-archival unless the site explicitly guarantees it.
- Preserve failed logs and manifests; repair from evidence rather than deleting it.
- Confirm `sacct` states, exit codes, output manifests, and quality summaries before
  releasing dependent work.
- Pilot every new hardware/software domain before scaling an array.

For scheduler semantics, see the upstream
[Slurm documentation](https://slurm.schedmd.com/documentation.html).
