# Security policy

PDE-OBS is a research benchmark and local/HPC command-line application. It is
not an authentication service and does not intentionally collect credentials.
Its security boundaries and known limitations are documented in
[`docs/SECURITY_MODEL.md`](docs/SECURITY_MODEL.md).

## Supported versions

PDE-OBS has not made a public package or dataset release yet. Security fixes
are applied to the default branch while the project is pre-release. Once
tagged releases begin, this table will identify the supported release lines.

| Version | Supported |
| --- | :---: |
| `main` (pre-release) | yes |
| Tagged releases | none yet |

## Report a vulnerability privately

Do not open a public issue for a suspected vulnerability. Use GitHub's
[private vulnerability reporting form](https://github.com/ru1ch3n/PartialObs--PDEBench/security/advisories/new).
Include, when possible:

- the affected commit or version;
- the vulnerable command, input, or component;
- a minimal reproduction or proof of concept;
- the security impact and any known mitigations; and
- whether you want public credit or prefer anonymity.

The maintainers will acknowledge a report within 14 days, normally sooner.
They will validate the report, agree on disclosure timing with the reporter,
prepare tests and a fix, and publish an advisory when users need to take
action. Confirmed medium-or-higher severity issues are prioritized for a
timely fix. Reporters receive credit unless they request anonymity.

For ordinary bugs and enhancement requests, use
[GitHub Issues](https://github.com/ru1ch3n/PartialObs--PDEBench/issues).
