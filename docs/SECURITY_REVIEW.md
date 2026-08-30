# Security review record

## Current status

**Qualifying human review: pending.** Automated evidence has been collected,
but OpenSSF Gold also requires human review of the security requirements and
security boundary. This document must not be cited as a completed Gold review
until the sign-off section is filled by a real reviewer.

## Review scope

- threat model, architecture, trust boundaries, and assurance case;
- configuration parsing and include/override handling;
- release downloader, redirects, paths, TLS, and integrity checks;
- HDF5, checkpoint, and plugin trust decisions;
- atomic output, provenance, and fail-closed quality gates;
- dependency and GitHub Actions supply chain;
- release, vulnerability-response, and maintainer-access processes.

## Automated evidence available to the reviewer

- CI tests and coverage report;
- Ruff formatting and lint checks;
- CodeQL and dependency-review results;
- `pip-audit`, Dependabot, secret scanning, and push protection;
- Atheris/ClusterFuzzLite configuration-parser fuzzing;
- reproducible-build comparison.

Automated evidence can reveal implementation defects but cannot complete the
required design review.

## Human review procedure

1. Read `SECURITY_MODEL.md`, `ARCHITECTURE.md`, and `ASSURANCE_CASE.md`.
2. Trace each critical claim to code and tests.
3. Challenge trust-boundary assumptions and identify missing abuse cases.
4. Record every finding with severity, affected component, evidence, owner,
   and resolution or explicit risk acceptance.
5. Add regression tests for corrected implementation findings.
6. Sign off only after all high-severity findings are closed and residual
   risks are documented.

## Sign-off

| Field | Value |
| --- | --- |
| Reviewed commit | pending |
| Reviewer name/account | pending |
| Reviewer relationship to project | pending |
| Review date | pending |
| Finding record | pending |
| High-severity findings remaining | pending |

