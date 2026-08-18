# Numerical validation20 report

This report records the completed SeaWulf factor-validation campaign for Git
commit `c70c726e13b812764711a60ea6cbef2ad54afb4e`. It is a dataset-generation
report only: no model was trained.

## Protocol

- Campaign: `numerics-validation20-c70c726`
- Grid: 7 PDEs x 4 boundaries x 10 settings x 20 samples
- Total: 5,600 samples in 840 regime shards at 128 x 128
- Quality schema: `1.3`
- Frozen normalized PDE-loss gate: `0.05`
- Generation rule: solve and checksum the complete ground truth first; apply
  observation masks only when `BenchmarkDataset` reads a sample
- Routing rule: solver and saved-frame density may differ by PDE, boundary,
  setting, and regime; observation protocol is never a solver input

The 840 generation tasks and the dependency-chained strict aggregate job all
completed with exit code zero. Strict plan coverage and shard validation found
840 HDF5 shards, 5,600 samples, seven PDE families, 5,600 valid quality records,
zero missing or invalid records, and zero non-empty
`*.quality-failures.jsonl` files. The generated campaign occupied 27 GiB on
SeaWulf scratch.

That 27 GiB campaign preserved every dense audit frame (`T=33` through
`T=513`) and is retained as historical validation evidence. The subsequent
storage protocol separates quality cadence from release cadence: PDE loss is
still measured on a dense in-memory trajectory, while each temporal HDF5 row
stores 30 uniformly spaced exact frames plus the dense `quality_T` and selected
frame indices. A new 30-frame SeaWulf pass is required before replacing the
historical hashes below.

## Normalized PDE losses

| PDE | Samples | Mean | Maximum | Worst sample |
|---|---:|---:|---:|---|
| Darcy | 800 | 0.00002262 | 0.00033192 | `darcy/neumann/threshold_level_set/high/000004` |
| Poisson | 800 | 0.00001553 | 0.00008154 | `poisson/neumann/smooth_grf/high/000001` |
| Helmholtz | 800 | 0.00003100 | 0.00126776 | `helmholtz/periodic/threshold_level_set/low/000000` |
| Heat | 800 | 0.00000348 | 0.00004155 | `heat/neumann/low_frequency_fourier/low/000002` |
| Reaction--diffusion | 800 | 0.00257861 | 0.03826684 | `reaction_diffusion/periodic/multi_frequency_fourier/low/000003` |
| Burgers | 800 | 0.01279232 | 0.04900286 | `burgers/dirichlet/multi_frequency_fourier/low/000001` |
| Navier--Stokes | 800 | 0.01205351 | 0.04825938 | `navier_stokes/dirichlet/multi_frequency_fourier/low/000002` |

Every sample passed the frozen `0.05` gate. These values are discrete,
operator-matched quality diagnostics; they do not by themselves constitute an
independent convergence or publication validation. Accordingly the aggregate
records `status=pass`, while `publication_ready=false` remains intentional.

## Audit hashes

| Artifact | SHA-256 |
|---|---|
| generation plan JSONL | `9f8b4ce42869a4aedfe29fd58ff002480166d652569887bb45a78fae4f824fcb` |
| resolved generation YAML | `e68169de1046ae75a00f5bf9e5feed0be03d6d64dd885fc9609c555f2ab4c491` |
| `summary.json` | `318a0027f1cc260c8c51fd32521768349a7be60ddede8266071ce9459468e30b` |
| `summary.quality.json` | `420e8f07fd87e5f56151a9e32288a99db82675171048ffb51490b9730825debf` |
| `summary.quality.csv` | `85b40dfff40a4c5a3cc74e8eda72454726cc8cb6250d9f507d9a7fe9d6a2bf04` |

The source plan, generated arrays, and full reports remain on temporary
SeaWulf scratch and require an independent archival copy before the site purge
window.
