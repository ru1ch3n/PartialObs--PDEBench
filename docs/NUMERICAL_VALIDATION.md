# Numerical validation gate

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
- For bounded/obstacle Navier-Stokes, a curl residual reconstructed from stored
  velocity is only a partial diagnostic. Release evidence also needs a full
  momentum/pressure residual or hidden validated state, plus divergence,
  no-penetration, wall/obstacle constraints, and interface-convergence tests.
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

Until this gate is completed, use the checked-in solvers for smoke data, method
development, and orchestration tests only.
