# Observation-conditioned baseline and campaign protocol

This document freezes the proposed primary sparse-recovery comparison and its
compute-planning assumptions. The machine-readable sources are
`pdeobs.protocol.observation_training_contract()` and
`configs/campaign/core_observation_medium.yaml`.

The campaign is partly planning-only. It must not be described as a completed
leaderboard, and planning-only method rows must not be submitted as executable
jobs.

## 1. Primary rule

For the primary matched-mask IID table:

> Train each normal trainable operator baseline independently for every PDE
> family and every observation protocol. The training and evaluation masks
> must match, and checkpoints must not be reused across observation protocols.

All nine masks are deterministic views of the same immutable physical HDF5
records. Generate the physical dataset once; do not create nine copies.

The existing experiment that trains on random 3% and evaluates the other eight
masks remains a separate cross-mask-transfer/mask-OOD table. It does not replace
the primary matched-mask comparison.

## 2. Observation protocols

At 128 by 128, the frozen views have these realized counts:

| ID | Protocol | Observed cells | Realized fraction |
| --- | --- | ---: | ---: |
| O1 | random 1% | 164 | 1.0010% |
| O2 | random 3% | 500 | 3.0518% |
| O3 | random 5% | 819 | 4.9988% |
| O4 | random 10% | 1,638 | 9.9976% |
| O5 | regular grid | 441 | 2.6917% |
| O6 | missing block | 12,288 | 75.0000% |
| O7 | line sensors | 508 | 3.1006% |
| O8 | boundary sensors | 508 | 3.1006% |
| O9 | clustered sensors | 492 | 3.0029% |

Always report the realized count and fraction. Missing block is not
density-matched to the approximately 3% structured views, so differences
cannot be attributed only to observation geometry.

## 3. Method-specific fit and reuse rules

| ID | Method slot | Fit/reuse rule | Current status |
| --- | --- | --- | --- |
| M1 | RBF / interpolation | no training; evaluate every result cell | executable built-in |
| M2 | Gappy POD / PCA | fit one basis per PDE on training records only; reuse across masks | planning-only |
| M3 | mask-channel U-Net | one checkpoint per PDE and mask | executable compact reference |
| M4 | mask-channel FNO | one checkpoint per PDE and mask | executable compact reference |
| M5 | CNO | one checkpoint per PDE and mask | executable CNO-like reference |
| M6 | DeepONet | one checkpoint per PDE and mask | external adapter required |
| M7 | PINN or PINO | choose one frozen implementation; the campaign arithmetic assumes an amortized operator-level implementation | adapter and choice required |
| M8 | Transolver or GNOT | choose and freeze exactly one implementation | adapter and choice required |
| M9 | DiffusionPDE | freeze the exact upstream prior once per compatible PDE distribution; test all masks | external adapter/prior required |
| M10 | FunDPS | freeze the exact upstream prior once per compatible PDE distribution; test all masks | external adapter/prior required |

The ten rows are comparison slots, not ten available registry entries. The
Builder must not generate commands such as `pdeobs train --model deeponet`
until a real adapter is installed and reports compatible capabilities.

For DiffusionPDE and FunDPS, record the upstream repository and commit, license,
checkpoint digest, training distribution, whether the prior was pretrained or
retrained, and every inference/sampling seed. Do not tune a separate prior for
each mask under this protocol.

Gappy POD must fit only on the canonical training split. Using validation or
test samples to form the basis is leakage.

## 4. Dataset accounting

The physical factorization has seven PDEs, four boundary protocols, ten
settings, and 2,000 full-tier records per macro case.

| Tier | Total records | Records per PDE | Optimizer-training records per PDE |
| --- | ---: | ---: | ---: |
| medium | 140,000 | 20,000 | approximately 14,000; 13,916-14,016 for the frozen nested seed |
| full | 560,000 | 80,000 | exactly 56,000 |

The 80,000 and 20,000 values are complete per-PDE pools, not training-set
sizes. Validation and test rows must remain excluded from fitting.

Periodic Navier-Stokes stores one-channel vorticity while bounded and obstacle
cases store two-channel velocity. A single checkpoint across all four
boundaries is therefore not currently shape-compatible. Before claiming one
model per PDE, standardize the state representation or define and report
representation-specific checkpoints.

## 5. Result cells and preparation jobs

For one sparse-recovery task and IID split:

```text
10 method slots x 7 PDEs x 9 observation protocols = 630 result cells
```

For the six normal trainable slots:

```text
6 x 7 x 9 = 378 neural fits at one seed
```

If only the PINN/PINO slot uses three seeds, add `2 x 7 x 9 = 126` fits:

```text
378 + 126 = 504 neural fits
```

Gappy POD adds seven PDE-basis fits. DiffusionPDE and FunDPS add no
prior-training jobs when compatible frozen upstream checkpoints exist, or up
to `2 x 7 = 14` prior jobs when both must be trained.

| Seed/prior assumption | Total fit/preparation jobs |
| --- | ---: |
| one seed; compatible external priors | 385 |
| one seed; both external priors retrained per PDE | 399 |
| PINN/PINO three seeds; compatible external priors | 511 |
| PINN/PINO three seeds; both external priors retrained per PDE | 525 |

These are not raw evaluation-run counts or scheduler-job counts. Seeded methods
and stochastic diffusion inference create additional runs. Five result views
(IID plus four factor OOD views) would contain `10 x 7 x 9 x 5 = 3,150`
cells, but leakage-free OOD generally also requires separate training and is
not included in the quoted IID GPU budget.

For a final statistical comparison, use the same seed policy for every
stochastic learned baseline where practical. Giving three seeds only to
PINN/PINO is a screening plan, not a fair uncertainty protocol. Three seeds for
all six normal slots would require 1,134 neural fits.

## 6. Provisional A6000 capacity scenario

The following values are rough, unmeasured planning assumptions from the
proposal. They are not benchmark results, SLAs, or SeaWulf runtime predictions.

```text
12 GPUs x 10 days x 24 hours = 2,880 theoretical GPU-hours
75%-80% usable capacity       = 2,160-2,304 GPU-hours
```

| Scenario | Provisional GPU-hours | Conclusion before pilot |
| --- | ---: | --- |
| full, PINN/PINO one seed | 4,200-4,600 | outside ten-day capacity |
| full, PINN/PINO three seeds | 7,000-7,400 | outside ten-day capacity |
| medium, PINN/PINO three seeds | 1,800-2,300 | potentially feasible but tight |

The attachment's rounded usable budget was 2,100-2,300 GPU-hours. The exact
75%-80% arithmetic is shown above. Dividing runtime by four when moving from
full to medium is not a validated scaling law for per-instance PINNs,
fixed-cost prior training, diffusion sampling, or I/O. Run a measured pilot and
recompute the schedule.

SeaWulf currently uses a shared A100 partition, one GPU per launcher job, with
site queue and time limits. It does not guarantee twelve simultaneous GPUs.
Do not transfer the A6000 estimate directly to SeaWulf.

## 7. Recommended campaign presets

### Medium complete comparison

- Tier: medium (140,000 physical records).
- Task/split: sparse recovery, IID.
- Methods: all ten frozen slots after every required adapter is integrated.
- Masks: all nine.
- PINN/PINO planning seeds: three; use a consistent final seed policy.
- Result cells: 630.
- Neural fits: 504 under the special PINN/PINO seed rule.
- POD fits: 7.
- Prior jobs: 0-14.
- Total preparation: 511-525.

### Full hybrid anchor comparison

- RBF and Gappy POD: all nine masks.
- U-Net, FNO, CNO, DiffusionPDE, and FunDPS: random 1%, random 3%, and missing block.
- Result cells: `2 x 7 x 9 + 5 x 7 x 3 = 231`.
- Preparation jobs: `3 x 7 x 3 + 7 + 14 = 84` when both external priors are retrained.

A uniform three-mask alternative for all seven selected methods has 147 result
cells and the same maximum 84 preparation jobs.

The full hybrid table is a scale check, not a substitute for the medium primary
table. Neither is publishable until the numerical-validation and dataset-quality
release gates pass.

## 8. Ten-day execution order

1. Freeze adapters, method variants, masks, result schema, seeds, and exact Git
   revision. Run one PDE/three-mask pilot and measure runtime, memory, I/O,
   checkpoints, inference, and failures.
2. Generate the medium physical dataset once. Require exact-plan coverage,
   checksums, and the dataset-quality summary before any training dependency.
3. Run executable U-Net/FNO/CNO matched-mask cells in bounded scheduler windows.
   Integrate DeepONet only after its adapter tests pass.
4. Run the selected PINO/operator-level physics-informed adapter and
   Transolver/GNOT adapter. Reduce scope explicitly if the pilot invalidates the
   budget; never silently omit failed cells.
5. Evaluate RBF; fit leakage-free POD bases; run the frozen DiffusionPDE and
   FunDPS priors across masks.
6. Run the secondary mask-transfer/OOD table for the preregistered core subset.
7. Resume failed jobs, then freeze raw records, processed tables, configs,
   checkpoints, logs, unsupported/failure rows, and environment provenance.
8. Produce per-method, per-PDE, per-mask, runtime/memory, difficulty, and
   failure-case reports. Keep missing or unsupported cells explicit.

## 9. Quality management and result acceptance

Every physical sample must carry the versioned quality record described in
[QUALITY_CONTROL.md](QUALITY_CONTROL.md). Before training or evaluation:

1. validate the exact expected generation plan and every shard checksum;
2. require complete finite/geometry/initial/boundary checks;
3. report the available family-specific normalized PDE loss for each of the
   seven PDE rows, preserving missing families explicitly;
4. keep Helmholtz nominal residual separate from compact-transfer defect;
5. label bounded velocity-only Navier-Stokes residuals partial rather than
   fabricating a complete momentum residual;
6. retain rejected-sample `*.quality-failures.jsonl` records; and
7. archive `summary.json`, `summary.quality.json`, and
   `summary.quality.csv` with the plan, config, commit, and environment.

Dataset PDE losses measure the generated physical records. Model prediction
metrics and any validated prediction residuals are separate outputs; one must
not be substituted for the other.

## 10. Required run identity

Every fit/evaluation output path and result row must include at least:

```text
method implementation + upstream commit/checkpoint
PDE family + state representation
observation protocol + realized count/ratio + mask seed
IID/OOD split definition
training seed + inference/sampling seed
dataset manifest/checksum + quality summary
resolved config + Git commit + environment
status: completed / failed / unsupported / not run
```

This identity makes future method additions possible without changing the core
dataset and prevents missing or incompatible cells from being mistaken for
results.
