# Public engineering roadmap

This roadmap covers the public PDE-OBS software from August 2026 through
August 2027. Dates are targets, not promises. Scientific release gates remain
authoritative, and private experiment plans are intentionally out of scope.

## August-October 2026: harden the development baseline

- Enforce independent human review and contributor sign-off on new changes.
- Make wheel and source-distribution builds bit-for-bit reproducible in CI.
- Raise measured test coverage without excluding meaningful production code.
- Complete a documented human security review of the published threat model,
  trust boundaries, downloader, configuration parser, plugin surface, and
  release process.

## November 2026-January 2027: release readiness

- Exercise the signed-release procedure on a non-production release candidate.
- Publish a supported-version and upgrade policy with the first public release.
- Keep dependency, static-analysis, fuzzing, and secret-scanning findings at
  zero unresolved exploitable high-severity issues.
- Improve the quick start and extension examples using user feedback.

## February-May 2027: contributor and quality maturity

- Recruit and onboard a second release-capable maintainer.
- Maintain a queue of small, well-scoped contributor tasks.
- Reach at least 90% statement coverage and 80% branch coverage with useful
  tests, including regression tests for at least half of fixed defects.
- Expand fuzzing to another high-risk parser or file boundary.

## June-August 2027: operational resilience

- Rehearse maintainer-access continuity and release recovery.
- Re-run the human security review and close or explicitly accept every
  resulting finding.
- Review the architecture, assurance case, supported versions, dependencies,
  and public documentation for drift.

## Non-goals

This roadmap does not authorize publishing private datasets, checkpoints,
predictions, credentials, cluster details, or unpublished scientific results.
It does not weaken numerical, provenance, health, or release gates.

