# Numerical validation gate

The bundled PDE generators are compact development references. This checklist
defines the minimum gate before generated arrays can be called PDE-OBS paper
ground truth. Passing storage/unit tests is necessary but not sufficient.

## Required evidence

For every family, boundary, setting, and physical regime used in a release:

1. Compare at least three spatial resolutions and demonstrate the expected
   refinement trend against an over-resolved or independently implemented
   reference.
2. Report normalized discrete equation residuals and boundary-condition
   residuals, with thresholds fixed before producing the release.
3. Test all ten setting generators and difficult low-regularity cases, not only
   smooth random fields.
4. For temporal equations, show time-step refinement and monitor the applicable
   invariants or dissipation laws over all saved frames.
5. For bounded/obstacle Navier-Stokes, report divergence, no-penetration,
   no-slip/free-slip, solid-mask exclusion, energy, and enstrophy diagnostics.
6. Run the complete matrix at the paper resolution with finite-value, shape,
   deterministic-seed, and duplicate-ID checks.
7. Freeze the solver Git revision, dependency versions, resolved solver options,
   validation report, generation plan, and checksums together.

## Release acceptance

A release manifest should identify the validated solver implementation and link
its report. `pdeobs aggregate --validate-shards --expected-plan PLAN.jsonl`
checks storage integrity and plan completeness; it does not by itself prove
that a discretization is scientifically accurate.

Until this gate is completed, use the checked-in solvers for smoke data, method
development, and orchestration tests only.
