# OpenSSF Best Practices Passing readiness

This document maps the public PDE-OBS repository to the OpenSSF Best Practices
**Passing** criteria. It is an evidence index, not a badge claim. The badge must
not be displayed until the project record at
<https://www.bestpractices.dev/> has been reviewed and reports `passing`.

The private experiment repository, datasets, checkpoints, and prediction
artifacts are outside the public badge scope and must not be published merely
to satisfy a repository check.

| Passing area | Repository evidence | Readiness |
| --- | --- | --- |
| Description, obtain, use, feedback, contribute | `README.md`, `SUPPORT.md`, `CONTRIBUTING.md`, project website | ready after merge |
| FLOSS license | `LICENSE` (MIT), `THIRD_PARTY_NOTICES.md` | ready |
| Basic and interface documentation | `README.md`, `docs/DATA_SCHEMA.md`, `docs/EXTENDING.md`, CLI help/tests | ready |
| HTTPS sites and public version control | GitHub repository and GitHub Pages URLs | ready |
| Maintenance and searchable discussion | Git history and GitHub Issues | continuous operational evidence |
| Unique versions and release notes | `pyproject.toml`, `CHANGELOG.md`, `RELEASING.md` | ready for first release; no release yet |
| Bug reports and response archive | GitHub issue templates and Issues | response-rate attestation required |
| Private vulnerability reports and response | `SECURITY.md`, GitHub private reporting | private reporting must remain enabled; response SLA is continuous |
| Build, install, and automated tests | `pyproject.toml`, `CONTRIBUTING.md`, `.github/workflows/ci.yml` | ready when required checks pass |
| Tests for new functionality | written policy in `CONTRIBUTING.md` and repository regression tests | ready |
| Warnings and lint | Ruff configuration plus blocking CI | ready when CI passes |
| Secure-development knowledge | `docs/SECURITY_MODEL.md` documents relevant threats and controls | a primary developer must self-attest the knowledge criteria |
| Cryptographic practices | no custom cryptography; SHA-256 and platform TLS only, documented in security model | applicable controls documented |
| MITM-resistant delivery | downloader enforces HTTPS and rejects downgrade redirects; artifacts are hash-checked | ready when regression tests pass |
| Known vulnerabilities and secrets | dependency audit/review, Dependabot, CodeQL, secret scanning and push protection | continuous; all confirmed medium+ findings must be handled within policy |
| Static and dynamic analysis | CodeQL, pytest, and Atheris/ClusterFuzzLite on changes and schedules | ready when workflows pass |
| Reproducible dependencies | `uv.lock` records exact versions, sources, and SHA-256 artifact hashes; CI refuses stale resolution | ready when CI passes |

## Remaining actions outside source control

1. Merge this work through a pull request with all checks passing and an
   independent human approval.
2. Keep GitHub vulnerability alerts, Dependabot security updates, secret
   scanning, and push protection enabled.
3. Protect `main` against force-push and deletion and require the CI checks used
   for releases.
4. Create or update the PDE-OBS entry at <https://www.bestpractices.dev/>, link
   each answer to this evidence, and make the required human attestations about
   secure-development knowledge and response history.
5. Add the official badge to `README.md` only after the service reports the
   project at the Passing level.

Scorecard is tracked separately because it produces per-check scores from 0 to
10 and has no universal `passing` threshold. Its workflow is included as a
continuous diagnostic, not as a substitute for the Best Practices badge.
