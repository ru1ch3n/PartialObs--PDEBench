# PDE-OBS reference protocol

This document freezes the defaults implemented in `pdeobs` version 0.1. The
defaults are deliberately configurable: a paper release should publish the
resolved YAML, generated manifest, checksums, and Git commit together.

This numerical/data protocol is subordinate to the frozen
[benchmark-paper contract](BENCHMARK_PAPER.md). The manuscript contribution is
the controlled benchmark and its anchor analyses; new semantic-ID, large
world-model, and foundation-model method claims are out of scope.

## Factorized data space

The reference suite is the Cartesian product of:

- seven families: Darcy, Poisson, Helmholtz, heat, reaction-diffusion,
  Burgers, and Navier-Stokes;
- four family-conditioned boundary protocols: Dirichlet/no-slip,
  Neumann/free-slip, periodic, and mixed/Robin/obstacle;
- ten condition generators: three Gaussian random fields, two Fourier
  mixtures, Gaussian blobs, piecewise blocks, a level set, a dipole/vortex
  pair, and a front/ring/shock field;
- three parameter regimes: low, medium, and high.

The physical meaning and numerical ranges of a regime are family-specific and
are stored in every sample's metadata. There are 280 macro cases and 840
family-boundary-setting-regime nodes.

## Size and deterministic identity

Release tiers contain 5, 20, 100, 500, or 2,000 instances per macro case. A
sample ID is derived from `(family, boundary, setting, regime, instance)` and a
global seed. The tiers are nested: a sample in `tiny` has the same ID and
content in every larger tier.

Because 2,000 is not divisible by three, the full tier assigns 667 samples to
low, 667 to medium, and 666 to high. The generator performs the same balanced
round-robin allocation for smaller tiers and records the realized counts.

The IID split is deterministic within each macro case:

| Split | Fraction | Full samples per macro case |
|---|---:|---:|
| train | 70% | 1,400 |
| validation | 15% | 300 |
| test | 15% | 300 |

OOD labels are additional views of the same immutable records. They never
alter the canonical IID assignment.

## Canonical arrays

Each shard stores channels-last arrays:

```text
condition   [N, H, W, V_cond]
trajectory  [N, T, H, W, V_state]
geometry    [N, H, W, 1]
```

Static PDEs use `T=1`. Temporal release arrays store exactly `T=30` ordered,
uniformly spaced, exact solver states spanning the configured physical time.
Numerical-validation jobs may retain a denser trajectory only in memory while
computing the saved-field PDE residual. Metadata records `quality_T`,
`stored_frame_indices`, and exact `stored_time_values`; the dense audit frames
are not written to HDF5. All current
Navier-Stokes validation routes store one vorticity channel; periodic,
rectangular bounded, and obstacle cases record different registered velocity-
reconstruction operators. The geometry array stores walls/obstacles.

Data are written as independent compressed HDF5 shards. Each array task owns
one shard and first writes a temporary file; the final name appears only after
shape and finite-value validation. A sidecar completion record stores schema
version, stable generation identity, row count, and SHA-256 digest. Metadata are also
available as a per-shard CSV plus a JSON summary for tools that should not open
the arrays. Prediction evaluation additionally emits per-sample metric JSONL
for difficulty and failure analysis.

Every generated sample also carries a versioned dataset-quality record. Shard
and dataset summaries report an explicit equation loss for every present PDE
family and preserve missing-family rows, so aggregation cannot hide incomplete
coverage. The default profile is report-only: an equation residual is a
measurement, not automatic acceptance. The loss definitions, profile semantics,
release gate, and known limitations are specified in
[Dataset quality control](QUALITY_CONTROL.md).

## Observations

The paper has nine deterministic observation views: random 1%, 3%, 5%, and
10%; regular grid; missing block; line sensors; boundary sensors; and clustered
sensors. At 128 by 128 their realized observed counts are 164, 500, 819, 1,638,
441, 12,288, 508, 508, and 492 respectively. Always report the realized count
and ratio. In particular, missing block observes 75% of the grid and is not a
density-matched comparison with the approximately 3% structured views.

Mask seeds are separate from PDE seeds, so all nine protocols are views of the
same immutable physical sample. Do not regenerate or store nine copies of the
140,000- or 560,000-record physical dataset.

### Primary matched-mask comparison

For the primary sparse-recovery IID table, every normal trainable operator
baseline is trained independently for each PDE family and each observation
view. The training and evaluation masks match, and a checkpoint is not reused
for a different mask. RBF requires no training; Gappy POD fits one basis per
PDE using only canonical training records; DiffusionPDE and FunDPS use one
frozen upstream prior per compatible PDE distribution and vary the observation
operator only at inference.

The default 500-point random mask remains the starter/reference configuration.
Training on that view and testing the other eight masks is a separate secondary
mask-transfer/OOD analysis, not a substitute for the nine matched-mask IID
rows. The full campaign rules and conditional job counts are in
[OBSERVATION_TRAINING_PROTOCOL.md](OBSERVATION_TRAINING_PROTOCOL.md).

## Official split views

- IID: new instances from seen factor combinations.
- Boundary OOD: leave one boundary protocol out.
- Setting OOD: hold out condition families, by default dipole and front.
- Parameter OOD: train on low/medium and evaluate the hard extrapolation regime.
- Combination OOD: hold out selected factor tuples while retaining each factor.
- Mask OOD: secondary cross-mask transfer--train with the 500-point random mask
  and test other ratios/geometries. Primary IID rows use matched-mask training.
- Time-horizon OOD: train short predictions and evaluate horizons 4 and 8.

## Numerical status

The bundled generators are compact, deterministic **development references**.
They are not approved to create paper ground truth. In particular,
reaction-diffusion is a scalar Allen-Cahn-like equation, Burgers is a scalar
two-dimensional transport equation, and the compact Helmholtz reference stores
a real-valued damped response. The Navier-Stokes difficulty regime changes both
viscosity and initial-vorticity scale; it is not a single-factor causal sweep.

The current validation candidates replace periodic-interior-plus-overwrite
bounded updates with boundary-in-operator sparse/FD/FV or bounded
vorticity--streamfunction routes. They have passed the complete one-sample
factor preflight but are not yet release ground truth. Before a scientific
release, pass and publish [the numerical validation gate](NUMERICAL_VALIDATION.md)
over the 20-sample-per-case campaign, add independent spatial/refinement
evidence, and freeze calibrated thresholds before generation. The PDE registry
permits replacing a case solver without changing storage, observation masks,
split views, or benchmark methods.
