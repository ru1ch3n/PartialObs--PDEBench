# Extending PDE-OBS

PDE-OBS separates scientific implementations from runners. A new solver,
method, mask, metric, or split is registered under a stable name and selected
from YAML; the command-line application does not need another hard-coded
branch.

## Add a method inside this repository

1. Implement the method contract in `src/pdeobs/methods/`.
2. Declare its capabilities (recovery, rollout, retrieval, CPU/GPU, static or
   temporal) and register it with the method decorator.
3. Add a small YAML file under `configs/method/`.
4. Add shape, serialization, and one-batch training tests.
5. Run `pdeobs list --kind methods` and the smoke benchmark.

The built-in U-Net, FNO, and CNO-like modules are compact reference baselines.
They are not claimed as exact reproductions of every upstream training recipe.

## Add a method from another package

Third-party packages can expose an entry point without modifying this Git
repository:

```toml
[project.entry-points."pdeobs.methods"]
my_method = "my_package.pdeobs_plugin:register"
```

The referenced callable can return a method class or register one or more
implementations. PDE solvers use the analogous `pdeobs.pdes` group. Discovery
happens on first use of the corresponding extensible component; importing the
top-level package alone does not import third-party solver packages.

For a paper-data solver, register the family under its canonical name with
replacement enabled, record the upstream code/version and all tolerances in the
job options (`solver_fidelity` and `solver_version` are promoted into sample
metadata), and run the gate in `NUMERICAL_VALIDATION.md`. A plugin must use the
public setting registry rather than maintaining a second set of condition
generators.

## Reproducibility contract

Every new component should:

- accept an explicit seed or generator;
- avoid global mutable state;
- advertise supported tasks and array layouts;
- validate input and output shapes;
- serialize enough state for exact inference;
- add its effective parameters to sample or run metadata;
- fail clearly when an optional dependency is absent;
- include a CPU smoke test, even if production use requires a GPU.

Large datasets, checkpoints, and environments do not belong in Git. Commit the
code, small fixtures, configurations, release manifests, and checksums; stage
bulk artifacts in cluster project/scratch storage or a dataset repository.
