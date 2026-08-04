# PDE-OBS reference protocol

This document freezes the defaults implemented in `pdeobs` version 0.1. The
defaults are deliberately configurable: a paper release should publish the
resolved YAML, generated manifest, checksums, and Git commit together.

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

Static PDEs use `T=1`. The reference temporal generators retain trajectories
for heat, reaction-diffusion, Burgers, and Navier-Stokes; the paper protocol
uses nine states. Periodic Navier-Stokes stores vorticity. Bounded and obstacle
cases store two velocity channels and an obstacle/wall mask.

Data are written as independent compressed HDF5 shards. Each array task owns
one shard and first writes a temporary file; the final name appears only after
shape and finite-value validation. A sidecar completion record stores schema
version, stable generation identity, row count, and SHA-256 digest. Metadata are also
available as CSV/JSONL for tools that should not open the arrays.

## Observations

The main 128-by-128 training view uses exactly 500 observed spatial locations.
Official evaluation views include random 1%, 3%, 5%, and 10%, regular-grid,
missing-block, line, boundary-only, and clustered sensors. Mask seeds are
separate from PDE seeds so the same physical sample can be evaluated under
multiple observation protocols without regeneration.

## Official split views

- IID: new instances from seen factor combinations.
- Boundary OOD: leave one boundary protocol out.
- Setting OOD: hold out condition families, by default dipole and front.
- Parameter OOD: train on low/medium and evaluate the hard extrapolation regime.
- Combination OOD: hold out selected factor tuples while retaining each factor.
- Mask OOD: train with the main random mask and test other ratios/geometries.
- Time-horizon OOD: train short predictions and evaluate horizons 4 and 8.

## Numerical status

The bundled generators are compact, deterministic **development references**.
They are not approved to create paper ground truth. In particular,
reaction-diffusion is a scalar Allen-Cahn-like equation, Burgers is a scalar
two-dimensional transport equation, and the compact Helmholtz reference stores
a real-valued damped response. The Navier-Stokes difficulty regime changes both
viscosity and initial-vorticity scale; it is not a single-factor causal sweep.

Boundary enforcement in the compact temporal/Helmholtz implementations is an
approximation and the bounded obstacle-flow reference is not yet backed by the
required divergence/no-penetration convergence evidence. Before a scientific
release, pass and publish [the numerical validation gate](NUMERICAL_VALIDATION.md)
against trusted solvers. The PDE registry permits replacing a generator without
changing storage, masks, split views, or benchmark methods.
