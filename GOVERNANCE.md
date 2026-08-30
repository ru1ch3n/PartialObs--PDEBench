# Project governance

PDE-OBS uses a maintainer-led governance model. The model applies to the
public software, documentation, and release metadata in this repository. It
does not grant authority over private datasets, unpublished experiments, or
external computing accounts.

## Roles and responsibilities

| Role | Responsibilities | Current holder |
| --- | --- | --- |
| Project lead | Sets public scope and roadmap, resolves disputes, appoints maintainers, and has final responsibility for releases. | `@ru1ch3n` |
| Maintainer | Triages issues, reviews pull requests, keeps continuous integration healthy, and may merge changes after policy gates pass. | vacant |
| Release manager | Verifies a clean reproducible build, signs release material, publishes release notes, and coordinates supported-version updates. | `@ru1ch3n` |
| Security coordinator | Receives private reports, coordinates fixes and disclosure, and maintains the threat model and security-review record. | `@ru1ch3n` |
| Contributor | Proposes focused changes, tests, reviews, documentation, or reproducible bug reports under the contribution policy. | open to all |

One person may hold several roles. Repository access by itself does not assign
a project role; assignments are recorded in this file through a reviewed pull
request.

## Decisions

Routine changes are decided in public issues and pull requests. Maintainers
seek technical consensus using reproducibility, compatibility, security, and
scientific-validity evidence. The project lead makes the final decision when
consensus is not reached and records the rationale publicly.

Security-sensitive matters stay private until coordinated disclosure is safe.
The security coordinator may embargo details, but the resulting fix still
requires tests and a public advisory or release note when users need to act.

## Change and review policy

- Changes reach `main` through pull requests with required checks passing.
- A human other than the author must approve a change before merge. Automated
  and AI reviews are useful supporting evidence but are not human approval.
- Reviewers follow [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md).
- User-visible changes update `CHANGELOG.md`; releases follow `RELEASING.md`.
- Private data, checkpoints, credentials, cluster configuration, and
  unpublished results remain outside this public repository.

## Appointing and removing maintainers

A maintainer candidate should have a sustained record of technically sound
contributions or reviews, understand the security model, use phishing-resistant
two-factor authentication where available, and agree to the Code of Conduct.
The project lead records an appointment or removal in a reviewed pull request.
Access is removed promptly when it is no longer needed.

## Continuity status

The project currently has one named release-capable maintainer. Therefore the
OpenSSF access-continuity and bus-factor requirements are **not yet satisfied**.
They become satisfied only after a second trusted maintainer has the knowledge
and actual platform permissions needed to triage, merge, respond to security
reports, and release without the project lead.

