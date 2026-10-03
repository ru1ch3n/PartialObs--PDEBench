# Running PDE-OBS on a Linux server

This guide covers a single Linux server reached over SSH. If the machine is a
managed cluster, use its scheduler instead of running long work on a login
node. Managed-cluster users should follow the portable
[Slurm HPC guide](../hpc/slurm/README.md).

## 1. Clone and install

Python 3.10 or newer, Git, and enough local storage are required.

```bash
git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git
cd PartialObs--PDEBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[train,test]"

export PDEOBS_DATA="$PWD/datasets"
export PDEOBS_RUNS="$PWD/runs"
mkdir -p "$PDEOBS_DATA" "$PDEOBS_RUNS"
pdeobs doctor
```

For an NVIDIA server, install a PyTorch build compatible with the machine's
driver/CUDA stack and then require the GPU check:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
pdeobs doctor --gpu
```

Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
when the normal `train` extra does not select the server's required build.

## 2. Run the complete smoke workflow

Start a persistent terminal before a long SSH session:

```bash
tmux new -s pdeobs
```

Inside that session:

```bash
source .venv/bin/activate
export PDEOBS_DATA="$PWD/datasets"
export PDEOBS_RUNS="$PWD/runs"

pdeobs generate --config configs/dataset/smoke.yaml \
  --output "$PDEOBS_DATA/smoke"
pdeobs aggregate --input "$PDEOBS_DATA/smoke" \
  --output "$PDEOBS_DATA/smoke/summary.json" \
  --validate-shards
pdeobs train --config configs/experiment/recovery_unet_smoke.yaml \
  --output "$PDEOBS_RUNS/smoke-train"
```

Detach with `Ctrl-b d`. After reconnecting to the server, recover the session
with `tmux attach -t pdeobs`. The smoke training checkpoint is written to
`$PDEOBS_RUNS/smoke-train/checkpoints/best.pt`.

The same server can use the config-free paper interface. Measure a small tier
before increasing `--num-workers`:

```bash
pdeobs protocol --check
pdeobs generate --tier signal --root "$PDEOBS_DATA" --num-workers 8
pdeobs train --task sparse_recovery --model fno \
  --data "$PDEOBS_DATA/pdeobs_signal" --split iid --mask random_3pct \
  --output "$PDEOBS_RUNS/fno-sparse-recovery"
pdeobs infer --task sparse_recovery --model fno \
  --ckpt "$PDEOBS_RUNS/fno-sparse-recovery/checkpoints/best.pt" \
  --data "$PDEOBS_DATA/pdeobs_signal" --split test
pdeobs eval --task sparse_recovery \
  --pred "$PDEOBS_RUNS/fno-sparse-recovery/preds.h5" \
  --data "$PDEOBS_DATA/pdeobs_signal" --metrics rel_l2,spectral
```

## 3. Run a strict train/validation/test example

The five-sample tiny tier is only a pipeline preflight. The focused signal-tier
configuration supplies 34 low-regime Poisson samples and includes all three IID
splits for the checked-in seed.

```bash
mkdir -p "$PDEOBS_DATA/plans"
pdeobs plan --config configs/dataset/recovery_signal.yaml --tier signal \
  --output "$PDEOBS_DATA/plans/recovery-signal.jsonl"
pdeobs generate --config configs/dataset/recovery_signal.yaml \
  --plan "$PDEOBS_DATA/plans/recovery-signal.jsonl" \
  --output "$PDEOBS_DATA/signal"
pdeobs aggregate --input "$PDEOBS_DATA/signal" \
  --output "$PDEOBS_DATA/signal/summary.json" \
  --validate-shards \
  --expected-plan "$PDEOBS_DATA/plans/recovery-signal.jsonl"

pdeobs train --config configs/experiment/recovery_unet.yaml \
  --output "$PDEOBS_RUNS/recovery-signal"

pdeobs eval --config configs/experiment/recovery_unet.yaml \
  --checkpoint "$PDEOBS_RUNS/recovery-signal/checkpoints/best.pt" \
  --output "$PDEOBS_RUNS/recovery-signal/metrics.json"

pdeobs infer --config configs/experiment/recovery_unet.yaml \
  --checkpoint "$PDEOBS_RUNS/recovery-signal/checkpoints/best.pt" \
  --output "$PDEOBS_RUNS/recovery-signal/predictions.h5"
```

To shorten a trial, append `--set training.epochs=2` to `pdeobs train`. To
resume an interrupted run, use the same output directory and its last
checkpoint:

```bash
pdeobs train --config configs/experiment/recovery_unet.yaml \
  --output "$PDEOBS_RUNS/recovery-signal" \
  --resume "$PDEOBS_RUNS/recovery-signal/checkpoints/last.pt"
```

## 4. Plan the nine-observation comparison

The primary IID table trains each normal operator baseline separately for every
PDE family and observation mask, while reusing the same physical dataset. The
random-3%-trained run above is a useful starter and the reference point for a
separate cross-mask-transfer/OOD study; it is not the complete matched-mask
table.

Use `configs/campaign/core_observation_medium.yaml` and
[OBSERVATION_TRAINING_PROTOCOL.md](OBSERVATION_TRAINING_PROTOCOL.md) to review
the full matrix before launching work. The ten-row suite contains methods that
are not yet registered in this package, so the campaign manifest is
planning-only until each external adapter passes its integration tests. Do not
submit a command for an unavailable method or treat a compact reference as an
exact upstream reproduction.

On a dedicated 12-A6000 server, the quoted ten-day capacity and GPU-hour values
are unmeasured planning assumptions. Begin with one PDE and three masks, record
actual samples/second, memory, checkpoint size, inference cost, and failure
rate, then recompute the schedule. A single all-boundary Navier-Stokes model
also requires an explicit canonical representation because periodic and
bounded records currently have different state channels.

## 5. Update from Git safely

Generated data, checkpoints, predictions, and runs are intentionally ignored by
Git. Commit your method/config changes before updating the checkout.

```bash
git status --short
git pull --ff-only
source .venv/bin/activate
python -m pip install ".[train,test]"
pdeobs doctor
```

For a reproducible campaign, record `git rev-parse HEAD`, copy the resolved YAML
and provenance files from the run directory, and export the environment. Keep a
second copy of valuable datasets and checkpoints outside the server.

## 6. Operational notes

- Confirm free space with `df -h` before generating larger tiers.
- Use `nvidia-smi` plus `pdeobs doctor --gpu` before scheduling GPU training.
- Watch memory/GPU use during the smoke run, then size production jobs from
  measurements rather than guesses.
- Do not run long work directly in an SSH shell without `tmux`, `screen`, or a
  scheduler.
- Do not publish data from the bundled compact solvers as paper ground truth
  until the [numerical validation gate](NUMERICAL_VALIDATION.md) is complete.
