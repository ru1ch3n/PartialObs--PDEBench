# ICLR 2027 title-and-appendix draft

This directory uses the official ICLR 2027 LaTeX style. At the current writing
stage, only the paper title and Appendix A--D are populated. The abstract and
main-text body are intentionally empty.

## Compile

Run from this directory:

```bash
tectonic --keep-logs --keep-intermediates main.tex
```

The appendix source is shared with `../appendix/pdeobs_appendix_A_D.tex`.

## SeaWulf figures

The figures in `figures/` were rendered from completed, accepted full-T15 HDF5
shards on SeaWulf. `figures/figure_snapshot.json` records the exact source
paths, rendering timestamp, mask counts, and accepted-shard quality summary.
The quality chart is a read-only snapshot, not the final strict aggregate.

`scripts/render_seawulf_figures.py` performs the read-only rendering, and
`scripts/render_seawulf_figures.sbatch` provides the one-CPU Slurm wrapper.
