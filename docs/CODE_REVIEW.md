# Code review standard

The project targets human review of every pull request and never less than the
OpenSSF Gold minimum of 50% of proposed modifications before release.

## Independence and acceptance

- At least one approving reviewer must be a person other than the author.
- Bot, static-analysis, and AI comments support review but do not count as the
  independent human approval.
- The latest material push must be reviewed; stale approvals are dismissed.
- Required CI, security, dependency, and conversation-resolution gates must
  pass before merge.
- The reviewer records approval, requested changes, or a reasoned comment in
  the pull request. Self-approval is not permitted.

## Reviewer checklist

The reviewer checks, as applicable:

1. the change is in public project scope and is the smallest coherent change;
2. behavior, interfaces, compatibility, and failure modes are documented;
3. untrusted inputs are allowlisted and failures remain fail closed;
4. secrets, private cluster details, datasets, checkpoints, predictions, and
   unpublished results are absent;
5. tests exercise success, rejection, and regression paths without weakening
   quality or scientific gates;
6. dependency, workflow-permission, serialization, network, path, and plugin
   risks are addressed;
7. generated files, release notes, provenance, and documentation stay current;
8. license and copyright statements remain present.

Security-sensitive changes also update the threat model or assurance case and
receive review from the security coordinator or another qualified reviewer.
Scientific-protocol changes require explicit evidence that factor coverage,
controls, metrics, and health thresholds were not weakened.

## Measuring the requirement

GitHub pull-request approvals are the record. Before each release, the release
manager checks every proposed modification since the previous release (or all
history for the first release) and records the reviewed fraction in the
release checklist. A release is blocked below 50%; the project target is 100%.

An urgent private security fix may be developed under embargo, but it still
requires non-author human review before the fixed release is published.

