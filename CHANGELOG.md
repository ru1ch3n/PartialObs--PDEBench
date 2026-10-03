# Changelog

This project follows [Semantic Versioning](https://semver.org/) for software
releases. Human-readable changes and publicly known vulnerabilities fixed by a
release will be listed here. The repository is currently pre-release; no
public package or dataset release has been made.

## Unreleased

### Added

- Rebuild the website as an evaluation and research platform: frozen paper
  snapshot, 3,969 filterable result blocks, exact CSV/JSON exports with provenance,
  seven transparent method cards, three proposed studies, releases and issue forms.
- Central website metadata cites arXiv:2609.36521v2. Paper/code project contents
  and settings remain independent of website publication.
- Source-audit verification scope, compute/coverage limitations and study-specific
  credit policy; preserve literature and historical tool URLs as secondary resources.
- Website mandatory PR approval and orphan signoff check removed at the owner's
  request; automated tests, dependency review, code analysis and fuzzing retained.
- Homepage link, image, stylesheet and archive-navigation regression checks.
- Current PDE-OBS public project homepage, with the previous homepage archived
  and links to the separate public code, preprint, data and model release.
- Public contribution, support, vulnerability-reporting, release, and OpenSSF
  readiness documentation.
- Dependabot, dependency review, CodeQL, and OpenSSF Scorecard automation.
- Least-privilege, immutable-SHA GitHub Actions configuration.
- Dependency auditing and coverage reporting in continuous integration.
- Hash-locked CI environments and continuous Atheris/ClusterFuzzLite testing
  of the public configuration parser.

### Security

- Update the hash-locked CI dependency urllib3 from 2.7.0 to 2.8.0 to address
  PYSEC-2026-4175, PYSEC-2026-4176 and PYSEC-2026-4177 reported by dependency audit.
- Reject insecure HTTP release manifests and artifact URLs, including HTTPS
  redirects that downgrade to HTTP.
- Raise the supported PyTorch and pytest dependency floors beyond versions
  currently associated with published advisories.
- Reject ambiguous non-string mapping keys and recursive configuration values
  before stable hashing or environment expansion.
