# PDE-OBS Appendix A--D artifact

This directory contains the standalone supplementary appendix for PDE-OBS.
Its organization follows the role of Sections A--D in the PDEBench
supplement: context, metric definitions, protocol details, and complete PDE
descriptions. The prose is newly written for PDE-OBS and does not reproduce
PDEBench text.

## Contents

- `pdeobs_appendix_A_D.tex`: complete English LaTeX source.
- `references.bib`: machine-readable bibliography used as a citation record.
- `figures/`: publication figures in vector PDF and raster PNG formats.
- `figure_data/`: exact CSV values behind the mask-count and validation plots.
- `generate_figures.py`: deterministic figure generator.
- `SOURCE_NOTES.md`: provenance boundaries and source mapping.

## Reproduce the figures

The publication figures currently included in this appendix were rendered on
SeaWulf from accepted full-T15 HDF5 shards. Each PDE is represented by its own
data card placed beside the corresponding numerical description. Their exact
sources, case metadata, and snapshot counts are recorded in
`figure_data/figure_snapshot.json`. The corresponding read-only plotting
source and one-CPU Slurm wrapper live in `../iclr2027/scripts/`.

For the older deterministic reduced-resolution explanatory figures, run:

From the repository root, with the project generation dependencies installed:

```bash
python paper/appendix/generate_figures.py
```

This local helper does not generate the full dataset and does not train any
model. It is retained for reproducibility of the earlier preflight artwork;
the ICLR draft uses the SeaWulf snapshot figures.

## Build the PDF

The source is compatible with Tectonic 0.17:

```bash
tectonic --keep-logs --keep-intermediates paper/appendix/pdeobs_appendix_A_D.tex
```

Run the command from `paper/appendix/`, or set that directory as the compiler
working directory so the relative figure paths resolve.
