# Problem-difficulty analysis

`pdeobs analyze` turns per-sample or per-run JSON/CSV metrics into deterministic,
plot-independent paper tables:

```bash
pdeobs analyze \
  --input runs/benchmark/metrics.json \
  --output runs/benchmark/difficulty.json \
  --config configs/analysis/difficulty.yaml
```

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
