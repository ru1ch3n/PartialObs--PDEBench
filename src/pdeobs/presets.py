"""Code-defined presets for the benchmark's one-line command interface.

The public CLI must keep working from an installed wheel, where repository-level
``configs/`` files are not necessarily present.  These compact presets therefore
live in Python.  YAML remains the advanced and fully customizable interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .generation import BOUNDARIES, PDE_FAMILIES
from .settings import SETTING_NAMES
from .splits import REGIMES, TIER_SIZES

TASK_ALIASES: Mapping[str, str] = {
    "sparse_recovery": "recovery",
    "sparse_to_full_recovery": "recovery",
    "recovery": "recovery",
    "forward": "forward",
    "forward_prediction": "forward",
    "inverse": "inverse",
    "inverse_prediction": "inverse",
    "world_modeling": "rollout",
    "world_model": "rollout",
    "rollout": "rollout",
    "semantic_retrieval": "retrieval",
    "retrieval": "retrieval",
    "solver_routing": "routing",
    "routing": "routing",
    "foundation_transfer": "transfer",
    "foundation_model_transfer": "transfer",
    "transfer": "transfer",
}

EXECUTABLE_FIELD_TASKS = ("recovery", "forward", "inverse", "rollout")
PROTOCOL_ONLY_TASKS = ("retrieval", "routing", "transfer")

SPLIT_ALIASES: Mapping[str, str] = {
    "iid": "iid",
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test",
    "boundary": "boundary",
    "boundary_ood": "boundary",
    "setting": "setting",
    "setting_ood": "setting",
    "parameter": "parameter",
    "parameter_ood": "parameter",
    "combination": "combination",
    "combination_ood": "combination",
}

MASK_ALIASES: Mapping[str, str] = {
    "random": "random_3pct",
    "random_1%": "random_1pct",
    "random_3%": "random_3pct",
    "random_5%": "random_5pct",
    "random_10%": "random_10pct",
    "1pct": "random_1pct",
    "3pct": "random_3pct",
    "5pct": "random_5pct",
    "10pct": "random_10pct",
}

_REFERENCE_FACTORS: Mapping[str, Mapping[str, str]] = {
    "recovery": {
        "pde": "poisson",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
    },
    "forward": {
        "pde": "poisson",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
    },
    "inverse": {
        "pde": "darcy",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
    },
    "rollout": {
        "pde": "heat",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "regime": "low",
    },
}


def normalize_task(task: str) -> str:
    """Normalize a paper-facing task name to the runner's canonical name."""

    key = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return TASK_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_ALIASES))
        raise ValueError(f"unknown benchmark task {task!r}; choose from {choices}") from exc


def normalize_mask(mask: str) -> str:
    key = str(mask).strip().lower().replace("-", "_").replace(" ", "_")
    return MASK_ALIASES.get(key, key)


def normalize_split(split: str) -> str:
    """Normalize public split names and reject config-only OOD protocols clearly."""

    key = str(split).strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"mask", "mask_ood", "time_horizon", "horizon_ood", "time_horizon_ood"}:
        raise ValueError(
            f"split {split!r} needs an explicit benchmark YAML because it defines an "
            "evaluation sweep rather than one dataset membership"
        )
    try:
        return SPLIT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown split {split!r}; choose from {', '.join(sorted(SPLIT_ALIASES))}"
        ) from exc


def default_generation_config(tier: str = "tiny") -> dict[str, Any]:
    """Return the canonical 7 x 4 x 10 generation design for one release tier."""

    selected = str(tier).strip().lower()
    if selected not in TIER_SIZES:
        raise ValueError(f"unknown tier {tier!r}; choose from {tuple(TIER_SIZES)}")
    return {
        "schema_version": 1,
        "name": "pdeobs-v0.1-reference",
        "seed": 20260804,
        "tier": selected,
        "resolution": 128,
        "trajectory_steps": 9,
        "dtype": "float32",
        "compression": "gzip",
        "compression_level": 4,
        "shard_size": 700,
        "samples_per_case": 2000,
        "families": list(PDE_FAMILIES),
        "boundaries": list(BOUNDARIES),
        "settings": list(SETTING_NAMES),
        "regimes": list(REGIMES),
        "tiers": dict(TIER_SIZES),
        "splits": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
            "regime_allocation_full": {"low": 667, "medium": 667, "high": 666},
        },
        "observations": {
            "train": {"protocol": "random", "count": 500},
            "evaluation": [
                {"protocol": "random", "ratio": 0.01},
                {"protocol": "random", "ratio": 0.03},
                {"protocol": "random", "ratio": 0.05},
                {"protocol": "random", "ratio": 0.10},
                {"protocol": "regular_grid", "ratio": 0.03},
                {"protocol": "block_missing", "missing_fraction": 0.25},
                {"protocol": "line_sensors", "ratio": 0.03},
                {"protocol": "boundary_sensors", "ratio": 0.03},
                {"protocol": "clustered", "ratio": 0.03},
            ],
        },
        "quality": {
            "enabled": True,
            "profile": "report",
            "require_pde_loss": True,
            "thresholds": {
                "finite_fraction_min": 1.0,
                "geometry_binary_max_error_max": 1.0e-6,
                "initial_condition_loss_normalized_max": 1.0e-6,
                "boundary_condition_loss_normalized_max": 1.0e-4,
                "pde_loss_normalized_max": None,
                "divergence_loss_normalized_max": None,
            },
        },
    }


def _method_config(task: str, model: str, *, channels: int = 1) -> dict[str, Any]:
    name = str(model).strip().lower().replace("-", "_")
    known: dict[str, dict[str, Any]] = {
        "unet": {
            "name": "unet",
            "kwargs": {"in_channels": channels, "out_channels": channels, "width": 32},
        },
        "fno": {
            "name": "fno",
            "kwargs": {
                "in_channels": channels,
                "out_channels": channels,
                "width": 32,
                "layers": 4,
                "modes": 12,
            },
        },
        "cno": {
            "name": "cno",
            "kwargs": {"in_channels": channels, "out_channels": channels, "width": 32},
        },
        "mae_small": {
            "name": "mae_small",
            "kwargs": {
                "in_channels": channels,
                "out_channels": channels,
                "width": 32,
                "latent_channels": 128,
                "patch_size": 8,
                "mask_ratio": 0.75,
                "preserve_visible": True,
            },
        },
        "zero": {"name": "zero", "kwargs": {}},
        "mean": {"name": "mean", "kwargs": {}},
        "nearest": {"name": "nearest", "kwargs": {}},
        "bilinear": {"name": "bilinear", "kwargs": {}},
        "rbf": {"name": "rbf", "kwargs": {}},
        "persistence": {"name": "persistence", "kwargs": {}},
    }
    if task == "rollout":
        if name == "persistence":
            return known[name]
        if name == "convlstm":
            return {
                "name": "convlstm",
                "kwargs": {"in_channels": channels, "hidden_channels": 32},
            }
        base = known.get(name, {"name": name, "kwargs": {}})
        return {"name": "autoregressive", "base": base}
    return deepcopy(known.get(name, {"name": name, "kwargs": {}}))


def build_experiment_preset(
    *,
    task: str,
    model: str,
    data: str | Path,
    split: str = "iid",
    mask: str = "random_3pct",
    factors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a strict, wheel-safe reference experiment mapping.

    The one-line preset intentionally selects one shape-compatible reference
    factor tuple.  Full cross-factor matrices remain explicit YAML benchmark
    suites so a model never silently mixes incompatible state representations.
    """

    canonical_task = normalize_task(task)
    if canonical_task in PROTOCOL_ONLY_TASKS:
        raise ValueError(
            f"{canonical_task!r} is defined as a lightweight benchmark-paper protocol, "
            "but it is not a field-regression Trainer task; use its dedicated utility "
            "API or an explicit plugin configuration"
        )
    if canonical_task not in EXECUTABLE_FIELD_TASKS:
        raise ValueError(f"task {canonical_task!r} has no executable preset")

    canonical_split = normalize_split(split)
    factor_overrides = dict(factors or {})
    selected_factors = dict(_REFERENCE_FACTORS[canonical_task])
    for key, value in factor_overrides.items():
        if key not in {"pde", "boundary", "setting", "regime"}:
            raise ValueError(f"unknown factor override {key!r}")
        if value:
            selected_factors[key] = str(value)

    factor_views = {"boundary", "setting", "parameter", "combination"}
    channels = 1
    state_representation: str | None = None
    if canonical_split in {"boundary", "setting", "parameter"}:
        held_out_key = {"boundary": "boundary", "setting": "setting", "parameter": "regime"}[
            canonical_split
        ]
        if held_out_key in factor_overrides:
            raise ValueError(
                f"--{held_out_key.replace('_', '-')} cannot be fixed while evaluating "
                f"{canonical_split}_ood"
            )
        selected_factors.pop(held_out_key, None)
    elif canonical_split == "combination":
        if canonical_task != "recovery":
            raise ValueError(
                "the built-in combination_ood one-line preset is available for "
                "sparse_recovery; use YAML for another task"
            )
        if any(key in factor_overrides for key in ("boundary", "setting")):
            raise ValueError(
                "combination_ood controls boundary and setting; do not fix either factor"
            )
        if factor_overrides.get("pde", "navier_stokes") != "navier_stokes":
            raise ValueError("the official combination_ood holdout belongs to navier_stokes")
        selected_factors = {
            "pde": "navier_stokes",
            "regime": str(factor_overrides.get("regime", "low")),
        }
        channels = 2
        state_representation = "velocity"

    data_config: dict[str, Any] = {
        "root": str(Path(data)),
        "train_glob": "**/*.h5",
        "verify_shards": True,
        "allow_split_fallback": False,
        "target_time": -1,
        "filters": selected_factors,
        "mask": {"protocol": normalize_mask(mask)},
    }
    if canonical_split in factor_views:
        data_config["ood_view"] = canonical_split
    else:
        data_config["split"] = canonical_split
    if state_representation is not None:
        data_config["state_representation"] = state_representation

    evaluation: dict[str, Any] = {"include_frequency_bands": True}
    if canonical_split in factor_views:
        evaluation["ood_views"] = [canonical_split]
    if state_representation is not None:
        evaluation["physical_representation"] = state_representation

    config: dict[str, Any] = {
        "schema_version": 1,
        "name": f"{canonical_task}-{model}-one-line",
        "seed": 20260804,
        "task": canonical_task,
        "data": data_config,
        "method": _method_config(canonical_task, model, channels=channels),
        "training": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "num_workers": 4,
            "checkpoint_every": 5,
            "device": "auto",
            "mixed_precision": True,
        },
        "evaluation": evaluation,
        "output": {"root": "runs"},
    }
    if canonical_task == "rollout":
        config["data"].update({"input_horizon": 1, "training_horizons": [1, 2]})
        config["training"]["batch_size"] = 8
        config["evaluation"].update({"horizons": [1, 2, 4, 8], "include_stability": True})
    return config


def build_benchmark_preset(
    name: str,
    *,
    tier: str,
    data: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a stable paper-facing benchmark preset into inline experiments."""

    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "fno_sparse_recovery": ("sparse_recovery", "fno"),
        "unet_sparse_recovery": ("sparse_recovery", "unet"),
        "cno_sparse_recovery": ("sparse_recovery", "cno"),
        "fno_world_modeling": ("world_modeling", "fno"),
    }
    try:
        task, model = aliases[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown benchmark preset {name!r}; choose from {', '.join(sorted(aliases))}"
        ) from exc
    selected_tier = str(tier).strip().lower()
    if selected_tier not in TIER_SIZES:
        raise ValueError(f"unknown tier {tier!r}; choose from {tuple(TIER_SIZES)}")
    data_root = Path(data) if data is not None else Path("data") / f"pdeobs_{selected_tier}"
    experiment = build_experiment_preset(
        task=task,
        model=model,
        data=data_root,
        split="iid",
        mask="random_3pct",
    )
    return {
        "schema_version": 1,
        "name": key,
        "tier": selected_tier,
        "output": {"root": "runs"},
        "experiments": [{"name": key, "config": experiment, "mode": "train_eval"}],
    }


def benchmark_preset_names() -> tuple[str, ...]:
    return (
        "cno_sparse_recovery",
        "fno_sparse_recovery",
        "fno_world_modeling",
        "unet_sparse_recovery",
    )
