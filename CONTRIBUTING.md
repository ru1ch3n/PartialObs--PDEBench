# Contributing to PDE-OBS

PDE-OBS welcomes reproducible bug reports, documentation improvements,
scientific validation, and well-tested code changes. By contributing, you
agree that your contribution is licensed under the repository's MIT license.

## Before proposing a change

- Search the [issue tracker](https://github.com/ru1ch3n/PartialObs--PDEBench/issues)
  for related work.
- Open a bug report for a reproducible defect or a feature request for a
  proposed change in scope.
- Report suspected vulnerabilities privately according to
  [`SECURITY.md`](SECURITY.md).
- Never commit datasets, checkpoints, predictions, credentials, private cluster
  configuration, or other generated run artifacts.

## Development environment

Python 3.10 or newer is required. A standard editable installation provides
the package, tests, linter, dependency audit, and optional research-site tools:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Run the same core checks used by continuous integration:

```bash
ruff format --check src tests
ruff check src tests
pdeobs protocol --check --config configs/dataset/default.yaml
python scripts/validate_papers.py
python scripts/generate_research_site.py
git diff --exit-code -- docs
pytest --cov=pdeobs
python -m pip_audit --local --skip-editable
```

Shell launchers must also pass `bash -n hpc/seawulf/*.sh
hpc/seawulf/*.sbatch` on a POSIX system.

## Coding and test policy

Python code must follow the Ruff configuration in `pyproject.toml`; CI treats
formatting and lint findings as failures. Public functions and externally
visible data formats should have concise documentation.

Every bug fix should include a regression test when practical. Major new
functionality must include automated tests for its success path, relevant
validation failures, and any new public interface. Changes to generated site
content must update the generator and the checked-in output together. Changes
to scientific protocols must preserve fail-closed validation and document the
reason for the change.

## Pull request process

1. Create a focused branch from `main`.
2. Make the smallest coherent change and update documentation and tests.
3. Run the checks above and include the results in the pull request.
4. Open a pull request that explains the problem, solution, scientific impact,
   and compatibility or security implications.
5. Address CI findings and review comments. Do not merge with failing required
   checks.

Release notes are maintained in [`CHANGELOG.md`](CHANGELOG.md). User-visible
changes should add an entry under **Unreleased**.
