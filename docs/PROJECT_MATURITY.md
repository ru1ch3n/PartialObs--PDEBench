# Project maturity evidence

This is a dated, evidence-based readiness record, not an OpenSSF badge claim.
Only the official service can award Passing, Silver, or Gold.

Audit date: **2026-08-30**. Audited public baseline: `873726e`.

## Verified strengths

- CI, CodeQL, dependency review/audit, Scorecard, and ClusterFuzzLite run with
  least-privilege permissions and immutable action SHAs.
- Dependencies are hash-locked; Dependabot, secret scanning, push protection,
  and private vulnerability reporting are enabled.
- The default branch blocks force-push/deletion, enforces administrators,
  requires status checks and conversation resolution, and uses linear history.
- The strengthened public test suite has 352 passing tests on this branch.
- K80/HPC evidence, private experiments, datasets, checkpoints, and predictions
  are deliberately outside this public maturity scope.

## Repository-side controls added or strengthened

| Area | Evidence | Status after this change |
| --- | --- | --- |
| Governance, roles, roadmap | `GOVERNANCE.md`, `MAINTAINERS.md`, `ROADMAP.md` | documented; second maintainer still required |
| Architecture and security requirements | `docs/ARCHITECTURE.md`, `docs/SECURITY_MODEL.md` | documented |
| Assurance argument | `docs/ASSURANCE_CASE.md` | documented with residual risks |
| Quick developer setup | `docs/QUICKSTART.md`, `uv.lock` | documented and locked |
| Code-review standard | `docs/CODE_REVIEW.md` | documented; historical human-review ratio still insufficient |
| Reproducible build | `docs/REPRODUCIBLE_BUILDS.md`, Hatchling, CI | local double-build passed for both sdist and wheel; merge-commit CI must also pass |
| Per-source-file notices | SPDX/copyright checker and source headers | must pass CI before claiming met |
| TLS floor | downloader context and tests | TLS 1.2 minimum, certificate verification retained |
| Dynamic analysis | ClusterFuzzLite/Atheris | active; required again before a major release |

## Conditions still preventing Gold

| Gold or prerequisite condition | Current evidence | Exact closure condition |
| --- | --- | --- |
| Official Passing and Silver prerequisites | No official badge record has been completed | Complete every applicable MUST criterion with truthful linked evidence on the official service |
| Access continuity and bus factor | One named release-capable maintainer | Add a second trusted maintainer with real issue, merge, security-response, and release capability; rehearse loss of either maintainer |
| Two unassociated significant contributors | GitHub reports one significant contributor | A second unassociated person must make a non-trivial contribution in the past year; a nominal account or approval is insufficient |
| Required 2FA | Account 2FA status is not publicly verifiable or project-enforced | Require 2FA for every developer who can change the central repository or access private reports, and retain evidence |
| Small contributor tasks | The label exists but no open issue uses it | Publish at least one real, bounded `good first issue`/small task |
| At least 50% non-author review | Recent merged pull requests have zero approvals | Obtain genuine human non-author approval before merge and maintain at least 50% reviewed proposed modifications before release; target 100% |
| Statement coverage >=90% | **75.14%** (6,922/9,212 statements) | Add useful tests until the full production-code measurement is at least 90%; do not omit hard modules to inflate it |
| Branch coverage >=80% | **60.95%** (1,931/3,168 branches) | Add useful true/false and error-path tests until at least 80% |
| Human security review in last five years | Automated checks only | A real human reviews the requirements and boundary, records findings, and completes `docs/SECURITY_REVIEW.md` |
| Hardened project website | GitHub Pages currently returns HSTS but not CSP, `X-Content-Type-Options: nosniff`, or `X-Frame-Options` | Serve the public site through a host/proxy that emits all required nonpermissive headers and verify the live responses |
| Signed widespread release | No public release exists | Before the first widespread release, publish signed artifacts/tags, verification instructions, and an offline-protected signing-key process |

The repository must not display an OpenSSF Gold badge until the official
project record reports Gold.
