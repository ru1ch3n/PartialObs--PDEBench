# Numerical solvers and validation gate

The dataset-generation phase is intentionally separated from every model-
training phase. No training job is part of the workflow on this page.

## Numerical protocol under validation

The previous compact bounded solvers applied a periodic Fourier interior step
and then overwrote boundary cells. That is not a consistent bounded-domain
discretization and is no longer the numerical protocol:

- periodic Navier--Stokes uses the FNO vorticity--streamfunction
  pseudospectral method, 2/3 dealiasing, a Crank--Nicolson viscosity update,
  and the reference solver's `1e-4` internal time step;
- periodic scalar diffusion uses its Fourier solution operator;
- bounded scalar elliptic/diffusion problems use second-order flux/finite-
  difference operators whose Dirichlet, Neumann, or Robin equations are part of
  the solve;
- Poisson and Darcy stop on a relative residual tolerance instead of a fixed
  Jacobi count;
- Helmholtz solves the nominal real BVP. It no longer substitutes the real
  part of a damped periodic transfer;
- rectangular no-slip and free-slip Navier--Stokes use a DST-diagonalized
  bounded streamfunction solve and wall-vorticity updates;
- obstacle/Robin Navier--Stokes uses a sparse streamfunction solve on the true
  fluid mask, Thom wall/obstacle vorticity, and SSP-RK2. It never applies a
  periodic update and then overwrites wall cells;
- nonlinear temporal solvers use substeps and record their Courant number,
  iteration counts, convergence residuals, and any clipping.

The FNO reference directly covers Burgers, Darcy, and periodic Navier--Stokes.
DiffusionPDE uses the same family of generators and adds second-order finite
differences for Poisson/Helmholtz. Our four-boundary, ten-setting matrix and
bounded-flow routes are explicit extensions and must therefore pass their own
convergence study; an upstream name is not validation by itself.

Solver choice and quality-audit cadence are explicit per-case protocol fields.
They may vary by PDE, boundary, setting, and regime, while every temporal HDF5
sample stores the same 30 uniformly spaced exact solver frames. Observation
masks are not solver inputs: every complete ground-truth field/trajectory is
generated and checksummed first, and masks are deterministic dataset views
applied later.
For example, the low/medium periodic and free-slip Navier--Stokes
multi-frequency strata use denser transient audit grids after the original
grid missed the frozen saved-frame residual gate. Same-seed seven-sample
SeaWulf refinements reduced the periodic maxima to `0.0380`/`0.0299`; the
free-slip maxima are `0.0463`/`0.0280`. The high regimes and other
topology-matched solvers remain unchanged.

The audit grids use `T = 1 + 29k` so each stored frame is an exact solver state
at a constant integer stride: no temporal interpolation is used. PDE loss is
computed before down-selection and retains the dense `quality_T` calibration
identity; only the 30 selected frames are persisted.

Primary references:

- [FNO data-generation code](https://github.com/ixScience/fourier_neural_operator/tree/master/data_generation)
- [FNO periodic Navier--Stokes solver](https://github.com/ixScience/fourier_neural_operator/blob/master/data_generation/navier_stokes/ns_2d.py)
- [DiffusionPDE official repository](https://github.com/jhhuangchloe/DiffusionPDE)
- [DiffusionPDE paper and data-generation appendix](https://openreview.net/pdf?id=z0I2SbjN0R)

No upstream source file is vendored. See `THIRD_PARTY_NOTICES.md` for license
and attribution boundaries.

## Two-stage SeaWulf gate

1. `configs/dataset/numerics_demo.yaml` exercises all seven PDEs on a small
   periodic smoke matrix.
2. `configs/dataset/numerics_validation20.yaml` covers all 280 PDE x boundary x
   setting macro cases at 20 samples each: 5,600 samples and 840 regime shards.

The 5,600-sample campaign is full **factor coverage**, not the 560,000-sample
paper tier. Its aggregate must report all seven normalized PDE losses,
boundary losses, divergence where applicable, missing/invalid quality counts,
and worst sample IDs. The true `full=2000` campaign remains blocked until the
report and refinement evidence are reviewed.

The complete SeaWulf campaign at commit `c70c726e13b` has now passed this
factor gate: 840/840 shards, 5,600/5,600 samples and quality records, all seven
PDE families, zero missing/invalid records, and zero non-empty failure logs.
The largest normalized losses were Burgers `0.04900286`, Navier--Stokes
`0.04825938`, and reaction--diffusion `0.03826684`; every sample remained below
the frozen `0.05` limit. See the checked-in
[validation20 report](../release/NUMERICS_VALIDATION20.md) for all seven PDEs,
worst sample IDs, and audit hashes. This is a passed full-factor numerical
quality preflight, not independent publication validation.

The earlier SeaWulf one-sample-per-stratum preflight covered all 840
PDE/boundary/setting/regime combinations at 128x128. After targeted saved-frame
refinement, the observed worst normalized PDE loss was below `0.05`; exact
first-transition replay remained about `1e-8`, bounded-flow divergence about
`1e-15`, and boundary losses about `1e-7` or smaller. These are preflight
results, not release evidence: the 5,600-sample run is required to test seed
variation before thresholds are frozen.

The bundled PDE generators are compact development references. This checklist
defines the minimum gate before generated arrays can be called PDE-OBS paper
ground truth. Passing storage/unit tests is necessary but not sufficient.

The automated [dataset quality-control report](QUALITY_CONTROL.md) is required
evidence for this gate, but it is not a substitute for convergence tests or an
independent reference solution. In particular, the default `report` profile is
designed to return warnings while PDE-loss thresholds remain unfrozen.

## Required evidence

For every family, boundary, setting, and physical regime used in a release:

1. Compare at least three spatial resolutions and demonstrate the expected
   refinement trend against an over-resolved or independently implemented
   reference.
2. Report normalized discrete equation residuals and boundary-condition
   residuals, using the versioned quality schema, with family/boundary/
   resolution thresholds fixed before producing the release.
3. Test all ten setting generators and difficult low-regularity cases, not only
   smooth random fields.
4. For temporal equations, show time-step refinement and monitor the applicable
   invariants or dissipation laws over all saved frames.
5. For bounded/obstacle Navier-Stokes, report divergence, no-penetration,
   no-slip/free-slip, solid-mask exclusion, energy, and enstrophy diagnostics.
6. Run the complete matrix at the paper resolution with finite-value, shape,
   deterministic-seed, and duplicate-ID checks.
7. Freeze the solver Git revision, dependency versions, resolved solver options,
   residual-operator/stencil version, active quality mask, threshold table,
   validation report, generation plan, and checksums together.

## Family-specific evidence

- For Helmholtz, distinguish the nominal equation residual from the compact
  solver's damped regularized-transfer defect. Reporting only the former does
  not validate the latter.
- For periodic Navier-Stokes, validate the vorticity residual, velocity
  reconstruction, divergence, energy, and enstrophy behavior.
- For bounded/obstacle Navier-Stokes, the stored state is native vorticity.
  The registered rectangular or masked streamfunction reconstruction measures
  the complete vorticity equation, discrete incompressibility, wall/obstacle
  constraints, and exact first-transition replay. Release evidence still
  requires spatial/interface refinement against an independent reference.
- For temporal families, distinguish the residual between saved frames from an
  integrator replay or local-truncation defect and report both when claiming
  time-integration accuracy.

## Release acceptance

A release manifest should identify the validated solver implementation and link
its report. `pdeobs aggregate --validate-shards --expected-plan PLAN.jsonl`
checks storage integrity and plan completeness; it does not by itself prove
that a discretization is scientifically accurate. The release must additionally
include a strict quality audit with all seven PDE families, calibrated thresholds,
and validated solver fidelities; see [Dataset quality control](QUALITY_CONTROL.md).

Until this gate is completed, generated data are numerical-validation artifacts,
not paper ground truth and not a public dataset release.
