# Quick start

These commands create a locked developer environment and run a small local
workflow without downloading a public dataset.

```bash
git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git
cd PartialObs--PDEBench
python -m pip install uv
uv sync --locked --extra dev --no-build-isolation

uv run pdeobs doctor
uv run pdeobs generate \
  --config configs/dataset/smoke.yaml \
  --output datasets/smoke
uv run pdeobs aggregate \
  --input datasets/smoke \
  --output datasets/smoke/summary.json \
  --validate-shards
uv run pdeobs list
```

Run the contributor checks with:

```bash
uv run ruff format --check src tests scripts/check_*.py .clusterfuzzlite/fuzz_config.py
uv run ruff check src tests scripts/check_*.py .clusterfuzzlite/fuzz_config.py
uv run python scripts/check_spdx_headers.py
uv run pytest --cov=pdeobs --cov-branch --cov-report=json:coverage.json
uv run python scripts/check_coverage_thresholds.py coverage.json --min-statements 75 --min-branches 60
uv run python -m pip_audit --local --skip-editable
```

Commits proposed for merge also need a DCO sign-off (`git commit -s`) and an
approval from a human reviewer other than the author; see
[`../CONTRIBUTING.md`](../CONTRIBUTING.md).

The smoke output is generated data and must not be committed. For server and
scheduler execution, continue with [`SERVER.md`](SERVER.md) and
[`../hpc/seawulf/README.md`](../hpc/seawulf/README.md).
