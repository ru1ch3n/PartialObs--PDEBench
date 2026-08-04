# Configuration guide

All commands accept a YAML file and repeated `--set key.path=value` overrides.
Configurations can include other files with `include:` and can use environment
variables such as `${PDEOBS_DATA:-datasets}`.

Included mappings merge recursively. Set a child value to YAML `null` when a
consumer supports an empty/default value and an inherited mapping must be
cleared (for example `method.kwargs: null`).

The checked-in configurations are executable reference protocols:

- `dataset/default.yaml`: the complete seven-family, four-boundary, ten-setting protocol.
- `dataset/smoke.yaml`: a tiny local or SeaWulf preflight run.
- `method/*.yaml`: compact baseline architectures, including the residual-CNN
  retrieval/multitask encoder and MAE-style small reconstruction anchor.
- `experiment/recovery_*.yaml`: sparse-to-full recovery training runs.
- `experiment/inverse_darcy_unet.yaml`: sparse solution-to-coefficient inversion.
- `experiment/recovery_unet_smoke.yaml`: one-epoch training on the 16x16 smoke data.
- `experiment/rollout_*.yaml`: autoregressive temporal runs.
- `experiment/pretrain_mae_small.yaml`: lightweight masked-reconstruction
  pretraining with the standard recovery runner.
- `experiment/rollout_navier_stokes_boundary_ood.yaml`: unified-velocity,
  leave-obstacle-boundary-out training with physical rollout metrics.
- `experiment/benchmark_smoke.yaml`: an evaluated baseline suite with a leaderboard.
- `analysis/difficulty.yaml`: metric direction, grouping, and failure-ranking defaults.
- `cluster/seawulf.yaml`: shared cluster paths and safe resource defaults.

For example:

```bash
pdeobs generate --config configs/dataset/smoke.yaml --output datasets/smoke
pdeobs train --config configs/experiment/recovery_unet_smoke.yaml
pdeobs benchmark --config configs/experiment/benchmark_smoke.yaml
```

Resolved configurations and a Git/environment provenance record are copied into
every run directory.
