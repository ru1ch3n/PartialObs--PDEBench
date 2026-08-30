# Security model

## Scope and trust boundaries

PDE-OBS is a local and HPC-oriented Python application. It has no built-in
authentication service, database server, or long-running network listener.
The main trust boundaries are:

1. command-line arguments and YAML/JSON configuration supplied by a user;
2. release manifests and artifacts downloaded from a publisher;
3. HDF5 datasets, checkpoints, and third-party plugins opened by a run;
4. the local filesystem and scheduler environment where outputs are written;
5. the source and dependency supply chain used to install PDE-OBS.

The primary protected assets are host integrity, filesystem contents,
credentials present in the process environment, scientific-result integrity,
and reproducibility evidence.

## Threats and controls

- **Configuration injection:** YAML is parsed with `yaml.safe_load`, public
  factor names are normalized and validated, and invalid schemas fail closed.
- **Path traversal and partial artifacts:** release paths are resolved beneath
  the selected destination; partial files remain distinct from completed
  files; atomic completion and locks prevent partial output from being treated
  as accepted evidence.
- **Delivery tampering:** remote manifests and artifacts must use HTTPS and
  redirect downgrades are rejected. Every artifact must match its declared
  SHA-256 digest and optional size before it is promoted from `.partial`.
  A digest authenticates an artifact only when the manifest itself comes from
  a trusted HTTPS publisher or a trusted local file.
- **Unsafe serialization:** configs and reports use YAML/JSON. Checkpoint
  loading is restricted to weights-only data where supported and is covered by
  tests. Untrusted HDF5 files and Python plugins are not sandboxed and should
  be treated as executable-risk inputs.
- **Command execution:** Git provenance uses a fixed argument vector without a
  shell. Third-party Python entry-point plugins execute code with the user's
  privileges and must be installed only from trusted sources.
- **Secret exposure:** provenance records an allowlist of non-secret scheduler
  fields rather than dumping the environment. Generated data, local `.env`
  files, checkpoints, and cluster settings are excluded from version control.
- **Supply chain:** CI runs tests, linting, dependency audit, dependency review,
  and CodeQL. Workflow actions are pinned to immutable commits and receive
  only the permissions they require.

## Cryptography

PDE-OBS does not implement authentication, encryption, password storage, key
agreement, or random key generation. It uses Python's standard SHA-256
implementation for integrity identifiers and HTTPS/TLS implementations from
the Python and operating-system stacks for transport security. It does not
implement custom cryptographic algorithms.

## Security assumptions and limitations

- A local file manifest is trusted by the user who supplies it.
- A checksum from an untrusted source is not a signature.
- Scientific datasets and checkpoints can consume substantial CPU, GPU,
  memory, and storage; resource limits must be enforced by the OS or scheduler.
- PDE-OBS does not sandbox plugins, Python environments, HDF5 libraries, or GPU
  drivers.
- Cluster account security, SSH keys, scheduler policy, and storage ACLs are
  deployment responsibilities outside the repository.

See [`SECURITY.md`](../SECURITY.md) for private reporting and response.
