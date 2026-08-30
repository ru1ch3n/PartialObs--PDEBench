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

### Security

- Reject insecure HTTP release manifests and artifact URLs, including HTTPS
  redirects that downgrade to HTTP.
- Raise the supported PyTorch and pytest dependency floors beyond versions
  currently associated with published advisories.
- Reject ambiguous non-string mapping keys and recursive configuration values
  before stable hashing or environment expansion.
