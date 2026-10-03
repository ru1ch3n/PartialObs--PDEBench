# Contributing to PDE-OBS

PDE-OBS welcomes reproducible bug reports, documentation improvements,
scientific validation, and well-tested code changes. By contributing, you
agree that your contribution is licensed under the repository's MIT license.
Participation is governed by the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Before proposing a change

Choose the repository that owns the change:

- Current benchmark code, protocols, paper and results:
  [ru1ch3n/PDE-OBS](https://github.com/ru1ch3n/PDE-OBS).
- Project homepage, literature index and archived tools: this repository.
  The website stays at its existing GitHub Pages URL and is published from
  `main:/docs`. Mandatory PR approval is disabled; required automated checks
  remain enabled. A website merge does not grant scientific verification status.
- Website changes link to existing public resources and may transcribe public
  result metadata into explicitly pinned snapshots. They never modify the public
  code repository. Confirm destinations exist before adding links.

The platform source is `scripts/build_platform.py`, with metadata in
`data/platform.json` and styles/interactions in `docs/assets/platform*` and
`docs/assets/results.js`. Run `python scripts/generate_research_site.py` to
regenerate all pages offline. Paper citation metadata follows arXiv v2 only in
this website; the canonical public project is not edited by the website build.

The frozen snapshot `docs/assets/snapshots/paper-20260925-v1.json` is a
full-precision transcription of a pinned public result index. Do not overwrite
it. Corrections and additions require a new version and release note. The
optional `scripts/import_paper_snapshot.py` reads an explicitly provided local
public source checkout and refuses to overwrite a destination; it never runs
experiments or modifies the source. Normal builds do not access that checkout.
Keep data/split, protocol, evaluator, software, checkpoint and result identities
separate. Preserve earlier tools and literature URLs as historical resources.

Use the Submit a Method, Reproduce a Result and Join a Study issue forms for
scientific contributions. Record training and inference permissions separately,
pretraining provenance (including unknowns), actual budgets, missing coverage
and precise verification scope. Check configuration, rescoring, inference and
training claims separately; never label a source import as a fresh rerun.
Accepted work is credited in release records. Authorship is study-specific and
requires substantive contributions, manuscript participation and accountability;
participation, compute donation or a PR does not guarantee authorship.

- Search the [issue tracker](https://github.com/ru1ch3n/PartialObs--PDEBench/issues)
  for related work.
- Open a bug report for a reproducible defect or a feature request for a
  proposed change in scope.
- Report suspected vulnerabilities privately according to
  [`SECURITY.md`](SECURITY.md).
- Never commit datasets, checkpoints, predictions, credentials, private cluster
  configuration, or other generated run artifacts.

## Development environment

Python 3.10 or newer is required. CI uses the checked-in `uv.lock`, including
artifact SHA-256 hashes, so contributors can reproduce the same complete
environment instead of resolving mutable dependency ranges:

```bash
uv sync --locked --extra dev --no-install-project
uv sync --locked --extra dev --no-build-isolation
```

Run the same core checks used by continuous integration:

```bash
uv run --no-sync ruff format --check src tests
uv run --no-sync ruff check src tests
uv run --no-sync pdeobs protocol --check --config configs/dataset/default.yaml
uv run --no-sync python scripts/validate_papers.py
uv run --no-sync python scripts/generate_research_site.py
git diff --exit-code -- docs
uv run --no-sync pytest --cov=pdeobs
uv run --no-sync python -m pip_audit --local --skip-editable
```

Shell launchers must also pass `bash -n hpc/slurm/*.sh
hpc/slurm/*.sbatch` on a POSIX system.

When dependencies change, regenerate `uv.lock` with the repository's pinned CI
version of uv and regenerate `.clusterfuzzlite/requirements.txt` from
`.clusterfuzzlite/requirements.in`. Review both diffs and their hashes before
committing them; CI rejects a stale lock file.

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
6. Obtain at least one independent approving review of the latest changes and
   resolve review conversations. New commits invalidate stale approvals.
   Administrators are subject to these protections; direct pushes, force pushes
   and deleting `main` are disabled. Do not use automatic cross-repository pushes
   to bypass review. A bot may propose a PR, but must not publish unreviewed changes.

Release notes are maintained in [`CHANGELOG.md`](CHANGELOG.md). User-visible
changes should add an entry under **Unreleased**.
