# Security assurance case

This assurance case connects PDE-OBS security requirements to design arguments
and repository evidence. It is a living engineering argument, not a claim that
all possible deployments or third-party plugins are secure.

## Scope and critical claims

### Claim 1: untrusted configuration fails closed

**Argument.** Configuration is parsed with `yaml.safe_load`, recursive or
ambiguous structures are rejected, registered names and numeric ranges are
allowlisted, and invalid inputs stop before work is scheduled or written.

**Evidence.** `src/pdeobs/config.py`, configuration and generation tests,
CodeQL, Ruff, and the Atheris/ClusterFuzzLite target.

### Claim 2: remote delivery resists downgrade and corruption

**Argument.** Remote sources must use HTTPS with certificate verification and
TLS 1.2 or newer. Redirects are checked again, paths are constrained beneath
the destination, and size plus SHA-256 must match before an artifact replaces
its `.partial` file.

**Evidence.** `src/pdeobs/download.py`, `tests/test_download.py`, release
manifest schema, and `docs/SECURITY_MODEL.md`.

**Limitation.** A digest does not authenticate a malicious manifest. Users
must obtain the manifest from a trusted HTTPS publisher or trusted local file;
signed public releases add stronger provenance.

### Claim 3: incomplete scientific artifacts are not accepted as complete

**Argument.** Atomic writes, completion records, checksums, plan identity,
factor coverage, and fail-closed aggregation separate partial work from
accepted evidence.

**Evidence.** storage, generation, aggregation, quality, provenance, and
regression-test modules and their tests.

### Claim 4: unsafe executable inputs are explicit trust decisions

**Argument.** YAML/JSON data paths avoid general object deserialization;
checkpoint loading uses weights-only behavior where supported. Python plugins,
HDF5 libraries, environments, and drivers are explicitly outside the sandbox
boundary and must be trusted or isolated by the operator.

**Evidence.** `docs/SECURITY_MODEL.md`, training tests for unsafe pickle
rejection, entry-point design, and deployment guidance.

### Claim 5: the public source supply chain is continuously checked

**Argument.** Minimal workflow permissions, immutable action SHAs, locked
dependencies, dependency review/audit, CodeQL, lint, tests, fuzzing, and secret
scanning reduce the chance that an unreviewed or vulnerable change reaches a
release.

**Evidence.** `.github/workflows`, `uv.lock`, `.clusterfuzzlite`, Dependabot,
branch protection, and `CONTRIBUTING.md`.

## Residual risk and acceptance

- HPC account security, scheduler isolation, filesystem ACLs, and device
  drivers are deployment responsibilities.
- Third-party plugins execute arbitrary Python with user privileges.
- Large inputs can exhaust compute or storage unless deployment limits exist.
- Public signed releases do not yet exist.
- A qualifying human security review is still required; automated analysis
  does not replace it.

Residual risks are reviewed before a public release. A new trust boundary,
network service, credential feature, or serialization format requires an
updated threat model, assurance argument, tests, and security review.

