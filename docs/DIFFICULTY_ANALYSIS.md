# Problem-difficulty analysis

`pdeobs analyze` turns per-sample or per-run JSON/CSV metrics into deterministic,
plot-independent paper tables:

```bash
pdeobs analyze \
  --input runs/benchmark/analysis_records.json \
  --output runs/benchmark/difficulty.json \
  --config configs/analysis/difficulty.yaml
```

`pdeobs benchmark` writes `analysis_records.json` and a flattened CSV beside
its leaderboard. Every evaluated IID, factor-OOD, mask-OOD, and time-horizon
row carries its method, task, factor context, observation protocol, split, and
metrics. The file is deterministic for a fixed set of experiment results and
is the supported hand-off to `pdeobs analyze`.

The JSON report contains overall statistics; separate summaries for observation
ratio, mask pattern, PDE, boundary, setting, and physical regime; rollout-horizon
and spectral tables; paired IID/OOD degradation; scaling levels and log-log
trends; and worst-case group/sample rankings. Non-empty tables are also written
as CSV files beside the JSON report. If `--output` has no `.json` suffix, it is
treated as a directory.

Nested metric mappings such as `{"metrics": {"rel_l2": 0.2}}` are accepted.
Common field aliases (`family`/`pde`, `mask_protocol`/`observation_pattern`,
`train_size`/`training_samples`, and others) are normalized automatically. For
unusual schemas, set `metrics`, `primary_metric`, or `higher_is_better` in the
analysis YAML. Positive degradation values always mean that OOD performance is
worse, regardless of metric direction.

For rollout experiments, configure the horizons exposed during optimization
separately from those reported at evaluation:

```yaml
data:
  training_horizons: [1, 2]
evaluation:
  horizons: [1, 2, 4, 8]
  ood_views: [time_horizon]
```

The largest training horizon controls the training target. Horizons outside
that set are evaluated as time-horizon OOD and are compared with the largest
seen horizon; they are never added to the optimization target.
