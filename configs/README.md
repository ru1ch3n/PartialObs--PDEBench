# Configuration guide

All commands accept a YAML file and repeated `--set key.path=value` overrides.
Configurations can include other files with `include:` and can use environment
variables such as `${PDEOBS_DATA:-datasets}`.

Included mappings merge recursively. Set a child value to YAML `null` when a
consumer supports an empty/default value and an inherited mapping must be
cleared (for example `method.kwargs: null`).

The checked-in configurations are executable reference protocols:

- `dataset/default.yaml`: the complete seven-family, four-boundary, ten-setting protocol.
  Its mandatory `quality` block embeds all seven family residual/loss reports;
  the default is report-only until scientific thresholds are calibrated.
- `dataset/numerics_demo.yaml`: a seven-PDE periodic numerical smoke test.
- `dataset/numerics_boundary_demo.yaml`: one small pass through every
  PDE/boundary pair using the registered topology-matched solver route.
- `dataset/numerics_validation20.yaml`: complete 280-factor coverage at 20
  samples per macro case (5,600 samples); this is not the paper full tier. Its
  dense `trajectory_steps_by_case` values are used only for in-memory PDE-loss
  measurement; `stored_trajectory_steps: 30` fixes the temporal HDF5 size.
  Solver mappings remain independent of observation masks.
- `dataset/numerics_full_t15.yaml`: the 560,000-sample, dataset-only SeaWulf
  campaign. It inherits the topology-matched numerical routes and strict PDE
  loss gate, stores `T=15` for temporal data (`T=1` for static data), keeps the
  denser audit path in memory, and limits each resumable shard to 200 samples.
- `dataset/smoke.yaml`: a tiny local or SeaWulf preflight run.
- `dataset/recovery_signal.yaml` and `dataset/rollout_signal.yaml`: focused
  34-sample cases with strict train/validation/test coverage for the default
  recovery and heat-rollout experiments.
- `method/*.yaml`: compact baseline architectures, including the residual-CNN
  retrieval/multitask encoder and MAE-style small reconstruction anchor.
- `experiment/recovery_*.yaml`: sparse-to-full recovery training runs.
- `experiment/forward_poisson_fno.yaml`: executable forward-prediction anchor.
- `experiment/inverse_darcy_unet.yaml`: sparse solution-to-coefficient inversion.
- `experiment/recovery_unet_smoke.yaml`: one-epoch training on the 16x16 smoke data.
- `experiment/rollout_*.yaml`: autoregressive temporal runs.
  Training horizons and evaluation horizons are separate; the checked-in
  protocol trains on 1/2 steps and evaluates 1/2/4/8 steps.
- `experiment/pretrain_mae_small.yaml`: lightweight masked-reconstruction
  pretraining with the standard recovery runner.
- `experiment/rollout_navier_stokes_boundary_ood.yaml`: unified-velocity,
  leave-obstacle-boundary-out training with physical rollout metrics.
- `experiment/benchmark_smoke.yaml`: an evaluated baseline suite with a leaderboard.
- `experiment/benchmark_paper_anchors.yaml`: paper field-anchor matrix with
  transparent/neural baselines and separately trained factor/mask OOD runs.
- `experiment/recovery_fno_{boundary,setting,parameter,combination}_ood.yaml`:
  leak-free factor-specific OOD training/evaluation configs.
- `experiment/recovery_fno_mask_ood.yaml`: 3% random training mask evaluated
  against every official ratio/pattern view. This is the secondary mask-transfer
  analysis, not the nine separately trained matched-mask IID table.
- `campaign/core_observation_medium.yaml`: machine-readable planning manifest
  for the ten-slot, seven-PDE, nine-observation medium-tier comparison. It is
  deliberately not an executable benchmark config: six requested external
  adapters are absent, and planning-only rows must never be submitted as jobs.
- `analysis/difficulty.yaml`: metric direction, grouping, and failure-ranking defaults.
- `cluster/seawulf.yaml`: documentation-only snapshot of the checked-in Slurm
  defaults. The launchers do not consume this YAML; use `sbatch` overrides or
  edit a copied launcher when a measured run needs different resources.

For example:

```bash
pdeobs generate --config configs/dataset/smoke.yaml --output datasets/smoke
pdeobs generate --config configs/dataset/rollout_signal.yaml --output datasets/signal
pdeobs train --config configs/experiment/recovery_unet_smoke.yaml
pdeobs benchmark --config configs/experiment/benchmark_smoke.yaml
pdeobs benchmark --config configs/experiment/benchmark_paper_anchors.yaml --dry-run
```

The config-free paper interface (`pdeobs generate --tier ...`, `train --task
... --model ...`, and `benchmark --preset ...`) uses code-defined presets so it
also works from an installed wheel. See
[`docs/BENCHMARK_PAPER.md`](../docs/BENCHMARK_PAPER.md). Retrieval, routing, and
foundation transfer remain lightweight APIs/protocols, not silently mapped onto
the field-regression Trainer.

Production experiment configs verify shard completion records and do not fall
back across missing splits. They use the signal tier because the five-sample
tiny tier cannot guarantee every split within one physical regime. The two smoke configs opt into split fallback
explicitly because their two samples cannot represent every official split.
Resolved configurations and a Git/environment provenance record are copied into
every run directory.
