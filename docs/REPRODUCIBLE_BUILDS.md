# Reproducible builds

PDE-OBS treats its source distribution and wheel as build artifacts. Hatchling
is the PEP 517 backend because its build targets support reproducibility and
`SOURCE_DATE_EPOCH`. The build environment is defined by `uv.lock`, the Python
version, the pinned `uv` version in CI, and the source commit.

CI builds both artifacts twice in separate directories with the same
`SOURCE_DATE_EPOCH` (the source commit time) and `PYTHONHASHSEED=0`, then
requires identical filenames and SHA-256 hashes. A mismatch fails the build.

To reproduce the check on a POSIX system:

```bash
uv sync --locked --extra dev --no-build-isolation
export SOURCE_DATE_EPOCH="$(git log -1 --pretty=%ct)"
export PYTHONHASHSEED=0
uv build --python .venv/bin/python --no-build-isolation --sdist --wheel --out-dir dist-a
uv build --python .venv/bin/python --no-build-isolation --sdist --wheel --out-dir dist-b
(cd dist-a && sha256sum *) > /tmp/pdeobs-build-a.sha256
(cd dist-b && sha256sum *) > /tmp/pdeobs-build-b.sha256
diff -u /tmp/pdeobs-build-a.sha256 /tmp/pdeobs-build-b.sha256
```

Rebuilding on a second machine is meaningful only when it uses the same
source, lock file, Python implementation/version, build frontend/version, and
environment variables. Release signatures authenticate published artifacts;
reproducibility independently checks their content.
