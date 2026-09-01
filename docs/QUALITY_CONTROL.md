# Dataset quality control

PDE-OBS computes a versioned quality record for every generated sample and
summarizes those records by shard, dataset, and PDE family. The report answers
whether the stored arrays are finite, structurally valid, and close to a stated
discrete equation. It is **not**, by itself, evidence that a solver is accurate
enough to create paper ground truth.

The bundled compact solvers are development references. Their quality records
are useful for finding regressions, corrupted shards, unstable parameter
regimes, and unusually large equation defects. They remain non-publication data
until the independent convergence and reference-solver gate in
[Numerical validation](NUMERICAL_VALIDATION.md) is completed.

## Stable report contract

Quality records use `schema_version: "1.3"` and contain only JSON-compatible
values. Every sample record includes:

- PDE family, boundary protocol, discrete operator label, and solver fidelity;
- finite-value fraction and binary-geometry error;
- initial- and boundary-condition losses where applicable;
- raw PDE residual MSE, residual RMS, normalization denominator, and normalized
  PDE loss;
- applicable physics diagnostics, including divergence, mass/energy change,
  trajectory growth, and solver stability counters;
- the configured thresholds, calibration context/hash, the result of each
  check, overall status, and a separate sample-level candidate-gate flag.

Dataset summaries keep an explicit `pde_losses` row for each of the seven
built-in families, including missing families. This prevents a successful
average over the present families from hiding incomplete benchmark coverage.
The per-family statistics report count, mean, standard deviation, minimum, and
maximum without loading all samples into memory.

The normalized residual convention is

```text
L_PDE = RMS(r) / (sum_j RMS(q_j) + epsilon),  epsilon = 1e-12,
```

where `r` is the discrete equation defect and the `q_j` are the signed equation
terms before they are combined. The report also preserves `MSE(r)` and
`RMS(r)`. The normalized value is dimensionless within one operator definition,
but it must not be compared across PDE families as if the discretizations and
units were identical.

## Family-specific equation losses

All derivatives are evaluated on the stored grid with the boundary convention
recorded by the sample. The geometry channel selects fluid cells for the PDE
loss. Thresholds therefore need separate calibration by family, boundary,
resolution, saved-frame interval, and solver implementation.

| Family | Reported discrete defect | Interpretation and required caution |
|---|---|---|
| Darcy | `-div(a grad(u)) - f` | Uses arithmetic face coefficients. The compact reference uses the identified fixed forcing `unit_square_sine_mix_v1`; an external solver must record its forcing identity or field. |
| Poisson | `-laplace(u) - f` | A static nominal-equation residual on the stored solution. |
| Helmholtz | `(-laplace - k^2)u - f` | This is the nominal real Helmholtz loss. The current validation candidate uses the boundary-specific sparse FD BVP operator. Legacy regularized-transfer shards retain a separate transfer defect and must not be mixed with this operator version. |
| Heat | periodic: `u[n+1] - exp(D dt laplace)u[n]`; bounded: `u_t - D laplace(u)` | The periodic FNO-style Fourier route uses its exact discrete heat-semigroup defect as the primary PDE loss and also reports the FD2 saved-frame strong form as an auxiliary metric. Bounded routes use their boundary-consistent FD2/CN operator. |
| Reaction-diffusion | `u_t - D laplace(u) - r(u-u^3)` | Measures the implemented scalar Allen-Cahn-like equation and also reports state-bound excess. |
| Burgers | `u_t + u(u_x+u_y) - nu laplace(u)` | Measures the implemented scalar two-dimensional transport equation. It is not a vector Burgers momentum residual. |
| Navier-Stokes | `omega_t + u omega_x + v omega_y - nu laplace(omega)` | All current validation routes store vorticity. Periodic velocity is reconstructed spectrally; rectangular bounded velocity uses a DST streamfunction; obstacle velocity uses the registered sparse masked streamfunction. The complete vorticity balance, divergence, boundary/obstacle defect, and exact first-transition replay are reported separately. |

The version 1.3 residual contract selects the measurement operator by solver
route; this prevents a spectral solution from being rejected merely because a
different FD stencil was used for auditing. The residual mask excludes the outer stencil cell for bounded
domains and excludes solid geometry plus a one-cell obstacle halo. Release
calibration must validate and freeze that mask together with the operators; it
must not tune the mask after seeing benchmark results.

## Constraint and stability diagnostics

The report complements the equation loss with checks that can expose failures a
small global residual might miss:

- `finite_fraction` covers condition, trajectory, and geometry arrays;
- `geometry_binary_max_error` measures distance from the binary set `{0, 1}`;
- `initial_condition_loss_normalized` is reported when condition and first
  trajectory state have the same shape;
- `boundary_condition_loss_normalized` measures the applicable Dirichlet,
  Neumann, Robin, no-slip/free-slip, or obstacle/inflow constraints;
- Navier-Stokes includes normalized divergence, kinetic-energy change, and
  enstrophy change;
- every trajectory includes mass change, state-energy change, growth factor,
  and maximum absolute value;
- solver-provided clip, Courant, substep, and substep-cap counters are propagated
  when available.

These quantities are diagnostics, not universal conservation laws. For
example, energy need not be constant in a forced, dissipative, or open-boundary
problem. Acceptance rules must be defined from the exact equation and boundary
protocol.

## Profiles and gates

PDE-OBS separates measurement from enforcement:

| Profile | Behavior | Appropriate use |
|---|---|---|
| `report` | Computes every available loss. An unfrozen PDE threshold yields `warning`, never an implicit pass. Malformed array/geometry contracts are quarantined. | Development, smoke generation, and threshold calibration. |
| `strict` | Rejects a sample when any configured check fails. A PDE threshold is enforced only when explicitly supplied. | Reproducible internal data production after thresholds are chosen. |
| `publication` | Reads a hashed per-stratum threshold table and solver-evidence attestation, but deliberately fails `independent_evidence_verification` because this package has no trust root/signature registry. | Expert preflight only; do not submit a generation campaign in this mode yet. |

`sample_quality_attestation_complete` means only that the self-reported hashes,
identity fields, and calibrated checks are internally consistent.
`sample_quality_gate_ready` and `publication_ready` intentionally remain false:
the repository cannot verify that referenced report/artifact hashes are genuine
or trusted. Publication additionally requires canonical full-factor plan
coverage, checksum validation, and independently verified release evidence.
Renaming the bundled compact solver's fidelity is not validation.

A typical reporting configuration is:

```yaml
quality:
  enabled: true
  profile: report
  require_pde_loss: true
  thresholds:
    finite_fraction_min: 1.0
    geometry_binary_max_error_max: 1.0e-6
    initial_condition_loss_normalized_max: 1.0e-6
    boundary_condition_loss_normalized_max: 1.0e-4
    pde_loss_normalized_max: null
    divergence_loss_normalized_max: null
```

The `null` values are deliberate: this repository does not invent scientific
acceptance thresholds before calibration.

## Auditing generated data

Generation stores the sample quality record with the sample metadata. A dataset
audit can use those stored records or recompute them from the arrays:

```bash
pdeobs quality --input data/generated --output quality-report.json
pdeobs quality --input data/generated --output quality-recomputed.json --recompute
```

Each completed shard publishes `*.quality.json`; strict aggregation writes
`summary.quality.json` and `summary.quality.csv`. The standalone command writes
the exact JSON path requested plus a same-stem CSV file.

A strict smoke audit can name every local requirement explicitly:

```bash
pdeobs quality \
  --input data/generated \
  --output quality-release.json \
  --strict \
  --max-pde-loss FROZEN_VALUE \
  --require-all-pdes \
  --require-validated-solvers
```

Use a calibrated family/boundary/resolution threshold table for a real release;
the single command-line maximum is mainly useful for conservative smoke gates.
The command above cannot establish publication-candidate readiness in the
current package because evidence hashes are unverified attestations. The web
Builder therefore blocks publication-mode generation instead of submitting a
known-failing local or Slurm campaign.
Keep the JSON report and its CSV projection beside the generation plan,
resolved configuration, shard manifests, checksums, code revision, and solver
validation report.

If a strict or publication-candidate sample fails during generation, it is not
accepted into the shard. Its identity, seed, metrics, thresholds, and failed
checks are retained atomically in `*.quality-failures.jsonl` for diagnosis.

## Freezing release thresholds

Before changing `report` data into publication data:

1. Select and version an independently validated solver for each family and
   boundary protocol.
2. Run spatial refinement for every static family and both spatial and temporal
   refinement for every temporal family.
3. Compare against an over-resolved or independently implemented reference and
   verify the appropriate conservation, dissipation, divergence, and boundary
   behavior.
4. Calibrate loss distributions on data that are disjoint from the benchmark
   test split, stratified by family, boundary, resolution, and physical regime.
5. Freeze thresholds, operator/stencil versions, active masks, solver revision,
   dependency lock, and acceptance protocol **before** generating the release.
6. Generate from the frozen plan, audit all seven families, and retain failed
   sample records rather than silently replacing them.
7. Publish the complete quality report and validation evidence with the dataset.

Threshold changes create a new quality-protocol version. They must never be
selected to improve a submitted model's benchmark score.

The current bounded Navier-Stokes validation representation stores native
vorticity and records the exact rectangular or masked velocity-reconstruction
operator. A future plugin may use a different state, but it must register and
version an equally complete residual, divergence, geometry, and boundary
contract; changing only a fidelity label is never sufficient.
