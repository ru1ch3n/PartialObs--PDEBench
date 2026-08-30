# Release process

PDE-OBS has not made a public package or dataset release. When the publication
and validation gates are complete, maintainers use this process:

1. Confirm that `main` is protected, required CI and security checks pass, and
   dependency audit findings are resolved or documented as non-exploitable.
2. Update the version in `pyproject.toml` using Semantic Versioning and move
   the relevant `CHANGELOG.md` entries from **Unreleased** to that version.
3. Build a wheel and source distribution from a clean checkout and smoke-test
   the installed wheel outside the checkout.
4. For dataset releases, validate the immutable release manifest, sizes, and
   SHA-256 hashes without adding data or model artifacts to Git.
5. Create a signed Git tag and publish human-readable release notes, including
   every fixed project vulnerability that had a public CVE or advisory ID.
6. Publish artifacts through HTTPS and provide signature and checksum
   verification instructions with the release.

Release tags and artifacts must never be cut from a dirty checkout. Private
signing keys must not be stored on the distribution service or in this
repository.
