# Changelog

This project follows [Semantic Versioning](https://semver.org/) for software
releases. Human-readable changes and publicly known vulnerabilities fixed by a
release will be listed here. The repository is currently pre-release; no
public package or dataset release has been made.

## Unreleased

### Added

- Public contribution, support, vulnerability-reporting, release, and OpenSSF
  readiness documentation.
- Dependabot, dependency review, CodeQL, and OpenSSF Scorecard automation.
- Least-privilege, immutable-SHA GitHub Actions configuration.
- Dependency auditing and coverage reporting in continuous integration.
- Hash-locked CI environments and continuous Atheris/ClusterFuzzLite testing
  of the public configuration parser.
- Public governance, maintainer, roadmap, architecture, assurance-case,
  code-review, reproducible-build, and maturity documentation.
- Automated DCO and per-source SPDX/copyright checks.
- Separate statement and branch coverage regression gates.
- Reproducible Hatchling source-distribution and wheel builds verified by
  independent SHA-256 comparisons in CI.

### Security

- Reject insecure HTTP release manifests and artifact URLs, including HTTPS
  redirects that downgrade to HTTP.
- Require certificate verification and TLS 1.2 or newer for HTTPS downloads.
- Raise the supported PyTorch and pytest dependency floors beyond versions
  currently associated with published advisories.
- Reject ambiguous non-string mapping keys and recursive configuration values
  before stable hashing or environment expansion.
