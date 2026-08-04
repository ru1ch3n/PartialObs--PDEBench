"""Deterministic local and Slurm-array dataset generation orchestration."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .registry import SETTING_REGISTRY, RegistryError
from .schema import GenerationSpec, Sample, derive_seed, json_safe, normalize_resolution
from .settings import SETTING_NAMES
from .splits import (
    REGIMES,
    build_split_plan,
    official_ood_labels,
    regime_counts,
    resolve_tier,
    tier_regime_counts,
)
from .storage import (
    AtomicHDF5ShardWriter,
    read_jsonl_manifest,
    write_jsonl_manifest,
)

PDE_FAMILIES = (
    "darcy",
    "poisson",
    "helmholtz",
    "heat",
    "reaction_diffusion",
    "burgers",
    "navier_stokes",
)


def _canonical_setting(name: str) -> str:
    """Resolve one registered setting name for stable IDs and safe paths."""

    try:
        SETTING_REGISTRY.get(name)
    except (RegistryError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown setting {name!r}") from exc
    return SETTING_REGISTRY.resolve_name(name)


BOUNDARIES = ("dirichlet", "neumann", "periodic", "robin_obstacle")
TEMPORAL_FAMILIES = frozenset({"heat", "reaction_diffusion", "burgers", "navier_stokes"})


def _regime_offset(regime: str, total: int) -> int:
    counts = regime_counts(total)
    if regime not in counts:
        raise ValueError(f"unknown regime {regime!r}")
    offset = 0
    for name, count in counts.items():
        if name == regime:
            return offset
        offset += count
    raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class GenerationJob:
    pde: str
    boundary: str
    setting: str
    regime: str
    sample_start: int
    sample_count: int
    shard_index: int
    output_path: str
    resolution: int | tuple[int, int] = 128
    seed: int = 0
    time_steps: int | None = None
    dtype: str = "float32"
    compression: str | None = "gzip"
    compression_level: int | None = 4
    tier: str = "full"
    macro_size: int = 2000
    options: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.pde not in PDE_FAMILIES:
            raise ValueError(f"unknown PDE family {self.pde!r}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"unknown boundary {self.boundary!r}")
        if self.regime not in REGIMES:
            raise ValueError(f"unknown regime {self.regime!r}")
        object.__setattr__(self, "setting", _canonical_setting(self.setting))
        if self.sample_start < 0 or self.sample_count < 1 or self.shard_index < 0:
            raise ValueError("sample_start/shard_index must be non-negative and count positive")
        object.__setattr__(self, "resolution", normalize_resolution(self.resolution))
        np.dtype(self.dtype)
        if self.compression_level is not None and int(self.compression_level) < 0:
            raise ValueError("compression_level must be non-negative")
        regime_limit = regime_counts(self.macro_size)[self.regime]
        if self.sample_start + self.sample_count > regime_limit:
            raise ValueError("job sample range exceeds its full regime allocation")
        object.__setattr__(self, "options", dict(json_safe(self.options or {})))
        object.__setattr__(self, "provenance", dict(json_safe(self.provenance or {})))

    @property
    def family(self) -> str:
        return self.pde

    @property
    def case_key(self) -> str:
        return "/".join((self.pde, self.boundary, self.setting))

    @property
    def job_id(self) -> str:
        return "/".join(
            (
                self.case_key,
                self.regime,
                f"shard-{self.shard_index:05d}",
            )
        )

    def sample_seed(self, regime_sample_index: int) -> int:
        return derive_seed(
            self.seed,
            self.pde,
            self.boundary,
            self.setting,
            self.regime,
            int(regime_sample_index),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pde": self.pde,
            "boundary": self.boundary,
            "setting": self.setting,
            "regime": self.regime,
            "sample_start": self.sample_start,
            "sample_count": self.sample_count,
            "shard_index": self.shard_index,
            "output_path": self.output_path,
            "resolution": list(normalize_resolution(self.resolution)),
            "seed": self.seed,
            "time_steps": self.time_steps,
            "dtype": self.dtype,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "tier": self.tier,
            "macro_size": self.macro_size,
            "options": dict(self.options or {}),
            "provenance": dict(self.provenance or {}),
            "job_id": self.job_id,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> GenerationJob:
        values = dict(row)
        values.pop("job_id", None)
        if "pde" not in values and "family" in values:
            values["pde"] = values.pop("family")
        if isinstance(values.get("resolution"), list):
            values["resolution"] = tuple(values["resolution"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    job_id: str
    output_path: str
    sample_count: int
    skipped: bool
    sha256: str


def build_job_grid(
    output_dir: str | Path,
    *,
    tier: str | int = "full",
    resolution: int | tuple[int, int] = 128,
    shard_size: int = 100,
    seed: int = 0,
    time_steps: int | None = None,
    dtype: str = "float32",
    compression: str | None = "gzip",
    compression_level: int | None = 4,
    families: Sequence[str] = PDE_FAMILIES,
    boundaries: Sequence[str] = BOUNDARIES,
    settings: Sequence[str] = SETTING_NAMES,
    regimes: Sequence[str] = REGIMES,
    macro_size: int = 2000,
    options: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    include_tier_dir: bool = True,
) -> list[GenerationJob]:
    """Expand the semantic product into independent, array-safe shard jobs."""

    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    tier_size = resolve_tier(tier, full_size=macro_size)
    tier_name = str(tier).lower() if isinstance(tier, str) else f"n{tier_size}"
    counts = tier_regime_counts(tier_size, full_size=macro_size)
    root = Path(output_dir)
    jobs: list[GenerationJob] = []
    canonical_settings = tuple(_canonical_setting(name) for name in settings)
    if len(set(canonical_settings)) != len(canonical_settings):
        raise ValueError("settings contain aliases that resolve to the same canonical setting")
    for factor_name, factor_values in (
        ("families", tuple(families)),
        ("boundaries", tuple(boundaries)),
        ("regimes", tuple(regimes)),
    ):
        if len(set(factor_values)) != len(factor_values):
            raise ValueError(f"{factor_name} contain duplicate semantic values")
    for family in families:
        for boundary in boundaries:
            for setting in canonical_settings:
                for regime in regimes:
                    count = counts[regime]
                    for shard_index, start in enumerate(range(0, count, shard_size)):
                        current_count = min(shard_size, count - start)
                        case_root = root / tier_name if include_tier_dir else root
                        output = (
                            case_root
                            / family
                            / boundary
                            / setting
                            / regime
                            / f"shard_{shard_index:05d}.h5"
                        )
                        jobs.append(
                            GenerationJob(
                                pde=family,
                                boundary=boundary,
                                setting=setting,
                                regime=regime,
                                sample_start=start,
                                sample_count=current_count,
                                shard_index=shard_index,
                                output_path=str(output),
                                resolution=resolution,
                                seed=seed,
                                time_steps=time_steps,
                                dtype=dtype,
                                compression=compression,
                                compression_level=compression_level,
                                tier=tier_name,
                                macro_size=macro_size,
                                options=options,
                                provenance=provenance,
                            )
                        )
    return jobs


def write_job_manifest(jobs: Iterable[GenerationJob], path: str | Path) -> Path:
    planned = list(jobs)
    job_ids = [job.job_id for job in planned]
    output_paths = [str(Path(job.output_path).resolve()) for job in planned]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("generation plan contains duplicate job IDs")
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("generation plan contains duplicate output paths")
    return write_jsonl_manifest((job.to_dict() for job in planned), path)


def load_job_manifest(path: str | Path) -> list[GenerationJob]:
    return [GenerationJob.from_dict(row) for row in read_jsonl_manifest(path)]


def resolve_array_index(index: int | None = None) -> int:
    """Use an explicit zero-based index or ``SLURM_ARRAY_TASK_ID``."""

    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("array index was not supplied and SLURM_ARRAY_TASK_ID is unset")
        index = int(value)
    if int(index) < 0:
        raise ValueError("array index must be non-negative")
    return int(index)


def select_array_job(
    jobs_or_manifest: Sequence[GenerationJob] | str | Path,
    index: int | None = None,
) -> GenerationJob:
    jobs = (
        load_job_manifest(jobs_or_manifest)
        if isinstance(jobs_or_manifest, (str, Path))
        else list(jobs_or_manifest)
    )
    selected = resolve_array_index(index)
    if selected >= len(jobs):
        raise IndexError(f"array index {selected} is outside 0..{len(jobs) - 1}")
    return jobs[selected]


def _sample_from_output(output: Any, metadata: Mapping[str, Any]) -> Sample:
    if isinstance(output, Sample):
        combined = dict(output.metadata)
        combined.update(metadata)
        return Sample(output.condition, output.trajectory, output.geometry, combined)
    try:
        condition = output.condition
        trajectory = output.trajectory
        geometry = output.geometry
    except AttributeError as exc:
        raise TypeError(
            "PDE generator output must be Sample-like with condition, trajectory, geometry"
        ) from exc
    combined = dict(metadata)
    combined["parameters"] = json_safe(getattr(output, "parameters", {}))
    diagnostics = json_safe(getattr(output, "diagnostics", {}))
    if diagnostics:
        combined["diagnostics"] = diagnostics
    return Sample(condition, trajectory, geometry, combined)


def generate_job(
    job: GenerationJob,
    *,
    resume: bool = True,
    overwrite: bool = False,
) -> GenerationResult:
    """Generate or resume one independent shard."""

    # Importing here keeps manifest inspection and ``--dry-run`` light-weight.
    from .pdes import generate_sample

    writer = AtomicHDF5ShardWriter(
        job.output_path,
        expected_count=job.sample_count,
        spec=job.to_dict(),
        resume=resume,
        overwrite=overwrite,
        compression=job.compression,
        compression_opts=job.compression_level,
    )
    if writer.completed:
        from .storage import read_shard_manifest

        manifest = read_shard_manifest(job.output_path)
        return GenerationResult(
            job.job_id,
            job.output_path,
            int(manifest["sample_count"]),
            True,
            str(manifest["sha256"]),
        )

    case_plan = build_split_plan(job.macro_size, seed=job.seed, case_key=job.case_key, shuffle=True)
    regime_offset = _regime_offset(job.regime, job.macro_size)
    try:
        for row in range(writer.count, job.sample_count):
            regime_index = job.sample_start + row
            macro_index = regime_offset + regime_index
            sample_seed = job.sample_seed(regime_index)
            output = generate_sample(
                family=job.pde,
                boundary=job.boundary,
                setting=job.setting,
                regime=job.regime,
                seed=sample_seed,
                resolution=normalize_resolution(job.resolution),
                # A dataset-wide trajectory setting applies only to temporal
                # equations; elliptic families are canonically T=1.
                time_steps=(1 if job.pde not in TEMPORAL_FAMILIES else job.time_steps),
                **dict(job.options or {}),
            )
            ood_labels = official_ood_labels(
                pde=job.pde,
                boundary=job.boundary,
                setting=job.setting,
                regime=job.regime,
            )
            state_representation = (
                "vorticity"
                if job.pde == "navier_stokes" and job.boundary == "periodic"
                else "velocity"
                if job.pde == "navier_stokes"
                else "scalar"
            )
            provenance = dict(job.provenance or {})
            git = provenance.get("git", {})
            metadata = {
                "sample_id": (
                    f"seed-{job.seed}/{job.pde}/{job.boundary}/{job.setting}/"
                    f"{job.regime}/{regime_index:06d}"
                ),
                "schema_version": "1.0",
                "pde": job.pde,
                "boundary": job.boundary,
                "setting": job.setting,
                "regime": job.regime,
                "state_representation": state_representation,
                "solver_fidelity": "compact_reference",
                "pdeobs_version": __version__,
                "resolution": list(normalize_resolution(job.resolution)),
                "regime_sample_index": regime_index,
                "macro_sample_index": macro_index,
                "split": case_plan[macro_index].split,
                "tier": job.tier,
                "seed": sample_seed,
                "generation_seed": job.seed,
                "config_hash": provenance.get("config_hash"),
                "git_commit": git.get("commit") if isinstance(git, Mapping) else None,
                **ood_labels,
            }
            writer.append(_sample_from_output(output, metadata).astype(job.dtype))
        manifest = writer.finalize()
    except BaseException:
        writer.close()  # retain *.partial for the next array resubmission
        raise
    return GenerationResult(
        job.job_id,
        job.output_path,
        int(manifest["sample_count"]),
        False,
        str(manifest["sha256"]),
    )


def run_array_job(
    manifest_path: str | Path,
    *,
    index: int | None = None,
    resume: bool = True,
) -> GenerationResult:
    return generate_job(select_array_job(manifest_path, index=index), resume=resume)


def jobs_from_spec(spec: GenerationSpec, output_dir: str | Path) -> list[GenerationJob]:
    """Split one explicit regime spec into shard-sized jobs."""

    jobs: list[GenerationJob] = []
    root = Path(output_dir).resolve()
    setting = _canonical_setting(spec.setting)
    for shard_index, start in enumerate(range(0, spec.num_samples, spec.shard_size)):
        count = min(spec.shard_size, spec.num_samples - start)
        path = (
            root
            / spec.tier
            / spec.pde
            / spec.boundary
            / setting
            / spec.regime
            / f"shard_{shard_index:05d}.h5"
        )
        if not path.resolve().is_relative_to(root):
            raise ValueError("generation output escapes the requested output directory")
        jobs.append(
            GenerationJob(
                pde=spec.pde,
                boundary=spec.boundary,
                setting=setting,
                regime=spec.regime,
                sample_start=start,
                sample_count=count,
                shard_index=shard_index,
                output_path=str(path),
                resolution=spec.resolution,
                seed=spec.seed,
                time_steps=spec.time_steps,
                dtype=spec.dtype,
                compression="gzip",
                compression_level=4,
                tier=spec.tier,
                # An explicit spec may be a smaller standalone case; ensure its
                # range is valid while retaining official full-size semantics.
                macro_size=max(2000, spec.num_samples * len(REGIMES)),
                options=spec.options,
            )
        )
    return jobs


def generate_from_spec(
    spec: GenerationSpec,
    output_dir: str | Path,
    *,
    resume: bool = True,
) -> list[GenerationResult]:
    return [generate_job(job, resume=resume) for job in jobs_from_spec(spec, output_dir)]


def _config_sequence(
    config: Mapping[str, Any], key: str, default: Sequence[str]
) -> tuple[str, ...]:
    value = config.get(key, default)
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(f"generation config {key!r} must be a list of names")
    result = tuple(str(item) for item in value)
    if not result:
        raise ValueError(f"generation config {key!r} cannot be empty")
    return result


def _configured_output(config: Mapping[str, Any]) -> Path:
    value = config.get("output", "datasets")
    if isinstance(value, Mapping):
        value = value.get("root", "datasets")
    return Path(str(value))


def jobs_from_config(
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    include_tier_dir: bool = True,
) -> list[GenerationJob]:
    """Adapt a resolved YAML mapping to the explicit generation job grid."""

    tier = config.get("tier", "full")
    configured_tiers = config.get("tiers")
    if isinstance(tier, str) and isinstance(configured_tiers, Mapping):
        configured_size = configured_tiers.get(tier)
        official_size = resolve_tier(tier, full_size=int(config.get("samples_per_case", 2000)))
        if configured_size is not None and int(configured_size) != official_size:
            raise ValueError(
                f"tier {tier!r} must contain {official_size} samples per macro case; "
                f"configuration requested {configured_size}"
            )
    compression = config.get("compression", "gzip")
    if compression in {"", "none", "None", False}:
        compression = None
    level = config.get("compression_level", 4 if compression else None)
    return build_job_grid(
        _configured_output(config) if output_root is None else output_root,
        tier=tier,
        resolution=config.get("resolution", 128),
        shard_size=int(config.get("shard_size", 100)),
        seed=int(config.get("seed", 0)),
        time_steps=config.get("trajectory_steps", config.get("time_steps")),
        dtype=str(config.get("dtype", "float32")),
        compression=None if compression is None else str(compression),
        compression_level=None if level is None else int(level),
        families=_config_sequence(config, "families", PDE_FAMILIES),
        boundaries=_config_sequence(config, "boundaries", BOUNDARIES),
        settings=_config_sequence(config, "settings", SETTING_NAMES),
        regimes=_config_sequence(config, "regimes", REGIMES),
        macro_size=int(config.get("samples_per_case", 2000)),
        options=config.get("solver_options", {}),
        provenance=config.get("_provenance", {}),
        include_tier_dir=include_tier_dir,
    )


def write_generation_plan(config: Mapping[str, Any], path: str | Path) -> list[GenerationJob]:
    """Write a manifest-driven Slurm array plan and return its ordered jobs."""

    jobs = jobs_from_config(config, include_tier_dir=True)
    write_job_manifest(jobs, path)
    return jobs


def _rebase_job(job: GenerationJob, output_root: str | Path) -> GenerationJob:
    """Place a manifest row below an explicit CLI output root."""

    output = (
        Path(output_root)
        / job.pde
        / job.boundary
        / job.setting
        / job.regime
        / f"shard_{job.shard_index:05d}.h5"
    )
    return replace(job, output_path=str(output))


def run_generation(
    config: Mapping[str, Any],
    output_root: str | Path,
    plan_path: str | Path | None = None,
    array_index: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """CLI adapter for a local tier or one manifest-selected array task.

    The returned mapping contains only JSON-compatible values so it can be
    printed directly or captured as scheduler provenance.
    """

    if plan_path is None:
        all_jobs = jobs_from_config(config, output_root=output_root, include_tier_dir=False)
    else:
        from .provenance import generation_identity

        loaded_jobs = load_job_manifest(plan_path)
        if not loaded_jobs:
            raise ValueError("generation plan is empty")
        identities = [generation_identity(job.provenance) for job in loaded_jobs]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("generation plan rows contain inconsistent code/config provenance")
        planned_identity = generation_identity(loaded_jobs[0].provenance)
        current_identity = generation_identity(config.get("_provenance", {}))
        if planned_identity != current_identity:
            raise ValueError(
                "generation plan code/config provenance differs from the current checkout; "
                "checkout the planned revision or regenerate the plan"
            )
        all_jobs = [_rebase_job(job, output_root) for job in loaded_jobs]
    if array_index is None:
        selected_jobs = all_jobs
    else:
        selected_jobs = [select_array_job(all_jobs, index=array_index)]

    summary: dict[str, Any] = {
        "status": "dry_run" if dry_run else "complete",
        "planned_job_count": len(all_jobs),
        "selected_job_count": len(selected_jobs),
        "array_index": array_index,
        "output_root": str(Path(output_root)),
        "plan_path": None if plan_path is None else str(Path(plan_path)),
        "force": bool(force),
        "sample_count": sum(job.sample_count for job in selected_jobs),
    }
    if dry_run:
        summary["jobs"] = [job.to_dict() for job in selected_jobs]
        return summary

    results = [generate_job(job, resume=not force, overwrite=force) for job in selected_jobs]
    summary["generated_job_count"] = sum(not result.skipped for result in results)
    summary["skipped_job_count"] = sum(result.skipped for result in results)
    summary["results"] = [json_safe(asdict(result)) for result in results]
    return summary
