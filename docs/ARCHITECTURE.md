# Architecture

PDE-OBS is a local and batch-oriented Python application. It has no persistent
server and no authentication subsystem.

## Major components

1. **CLI and configuration** — `pdeobs.cli` routes commands; `pdeobs.config`
   parses YAML with safe loading, resolves controlled includes, and validates
   overrides.
2. **Registries and protocol** — PDE families, methods, masks, settings, and
   metrics are selected through explicit registries. `pdeobs.protocol` checks
   the public benchmark contract for drift.
3. **Planning and generation** — `pdeobs.generation`, `pdeobs.routing`, and PDE
   solvers construct deterministic jobs and materialize numerical shards.
4. **Storage and datasets** — `pdeobs.storage` writes atomic HDF5 shards with
   manifests and checksums; `pdeobs.dataset`, `pdeobs.splits`, and
   `pdeobs.masks` expose controlled observation views.
5. **Methods and training** — classical and neural method adapters implement a
   common interface. Training and checkpoint code is local-process code and
   does not execute an external service.
6. **Evaluation and quality** — metrics, quality checks, aggregation, and
   reports fail closed when required evidence or factor coverage is missing.
7. **Operations** — `hpc/seawulf` maps immutable plans to scheduler jobs. The
   scheduler, filesystem, drivers, and account controls remain deployment
   boundaries rather than application components.

## Data flow

```text
configuration -> validated plan -> numerical shard -> immutable manifest
      |                                    |
      +-> registry selection               +-> dataset/mask/split view
                                                   |
                                                   +-> method/training
                                                           |
                                                           +-> evaluation
                                                               and reports
```

Each boundary records or validates identity information so a partial output,
different plan, or mismatched checkpoint cannot silently become accepted
evidence.

## Extension boundary

Third-party PDEs and methods may be loaded through Python entry points. A
plugin executes with the user's process privileges and is therefore trusted
code, not sandboxed data. Users should install plugins only from publishers
they trust and in an isolated environment.

## Network boundary

Normal generation, training, and evaluation are offline. The release
downloader is the only built-in network client. It accepts HTTPS (TLS 1.2 or
newer) and trusted local files, verifies certificates through the operating
system/Python trust store, rejects downgrade redirects, constrains output
paths, and verifies declared sizes and SHA-256 hashes.

The detailed threats, assumptions, and limitations are in
[`SECURITY_MODEL.md`](SECURITY_MODEL.md).

