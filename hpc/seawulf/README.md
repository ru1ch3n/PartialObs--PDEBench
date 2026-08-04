# Running PDE-OBS on SeaWulf

These launchers target Stony Brook's current SeaWulf Slurm environment. Verify
the queue and module names before a large campaign because cluster policy can
change.

## 1. Clone an exact Git revision

On `milan.seawulf.stonybrook.edu`:

```bash
module load slurm
git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git
cd PartialObs--PDEBench
git checkout <release-tag-or-commit>
```

Queued jobs do not pull from Git. This ensures every shard in a campaign uses
the revision recorded at submission.

## 2. Choose persistent and scratch locations

```bash
export PDEOBS_ENV=/gpfs/projects/<group>/envs/pdeobs
export PDEOBS_DATA=/gpfs/scratch/$USER/pdeobs/data
export PDEOBS_RUNS=/gpfs/scratch/$USER/pdeobs/runs
export PDEOBS_ARTIFACTS=/gpfs/projects/<group>/pdeobs/artifacts
```

Keep the environment and irreplaceable results in project space. Scratch is
appropriate for regenerable shards and temporary runs but is not a release
archive.

## 3. Build once in a compute allocation

Do not build packages on a login node. Start an interactive allocation, then:

```bash
bash hpc/seawulf/bootstrap.sh
"$PDEOBS_ENV/bin/python" -m pdeobs doctor --cluster seawulf --offline
```

`environment.yml` pins the reference PyTorch 2.5 / CUDA 12.4 environment. Set
`PDEOBS_CUDA_MODULE` if the selected SeaWulf architecture requires an explicit
CUDA module, and confirm the available module/driver combination before setup.
For a GPU preflight, set `PDEOBS_REQUIRE_GPU=1` inside a GPU allocation; the GPU
job scripts also stop immediately if PyTorch cannot see CUDA.

If compute nodes cannot access package servers, populate a compatible
wheelhouse in advance and set `PDEOBS_WHEELHOUSE` before running the bootstrap
script.

## 4. Run a single smoke task

```bash
mkdir -p logs
sbatch --array=0-0 hpc/seawulf/generate_array.sbatch \
  configs/dataset/smoke.yaml "$PDEOBS_DATA/smoke"
```

Inspect the output and `seff <job-id>` before submitting a larger array.
After that job succeeds, exercise the complete data-to-training path with the
resolution-aware smoke experiment:

```bash
mkdir -p logs
sbatch hpc/seawulf/train_gpu.sbatch \
  configs/experiment/recovery_unet_smoke.yaml \
  --output "$PDEOBS_RUNS/smoke-train"
```

## 5. Plan and submit a release tier

```bash
"$PDEOBS_ENV/bin/python" -m pdeobs plan \
  --config configs/dataset/default.yaml --tier tiny \
  --output "$PDEOBS_DATA/plans/tiny.jsonl"

bash hpc/seawulf/submit_generation.sh \
  "$PDEOBS_DATA/plans/tiny.jsonl" \
  configs/dataset/default.yaml \
  "$PDEOBS_DATA/tiny" \
  0 99
```

Each array element writes an independent shard. Array tasks never append to a
shared HDF5 or CSV file. After all tasks succeed, submit the aggregate/validate
step with a Slurm `afterok` dependency.

The checked-in plan has 840 independent regime-node jobs. SeaWulf may limit a
user to 100 queued jobs, and an array's pending elements can count toward that
limit. `submit_generation.sh` therefore accepts an inclusive `START STOP`
window and refuses windows larger than `PDEOBS_MAX_QUEUED_TASKS` (default 100).
Wait for `0 99` to finish, then submit `100 199`, and continue through the last
plan index; `%${PDEOBS_GENERATION_CONCURRENCY:-4}` in the array request limits
simultaneously running tasks. Confirm current site policy before changing either
limit.

The 560,000-sample full tier is approximately **296 GiB raw** before HDF5
compression. Compression depends strongly on the generated fields. Budget
additional scratch space for partial shards, logs, metadata, and any curated
copy to project storage; generate `tiny` first and measure its realized size
before reserving space for `full`.

## 6. Train and evaluate

```bash
mkdir -p logs
sbatch hpc/seawulf/train_gpu.sbatch configs/experiment/recovery_unet.yaml
sbatch hpc/seawulf/evaluate_gpu.sbatch \
  configs/experiment/recovery_unet.yaml /path/to/checkpoint.pt
```

The default A100 scripts request one GPU, eight CPU cores, 64 GB memory, and
eight hours. Change resources at submission time when measurements justify it;
variables are not expanded in `#SBATCH` headers.

Official references:

- [SeaWulf usage guidance](https://rci.stonybrook.edu/HPC/about/guidance)
- [queues](https://rci.stonybrook.edu/HPC/docs/architecture/queues-table)
- [Slurm jobs](https://rci.stonybrook.edu/HPC/docs/jobs/slurm-overview)
- [storage](https://rci.stonybrook.edu/hpc/faqs/requesting-storage-on-seawulf)
- [Conda environments](https://rci.stonybrook.edu/HPC/docs/software/conda)
