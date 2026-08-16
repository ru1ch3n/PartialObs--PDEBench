"""Frozen, machine-verifiable contract for the PDE-OBS benchmark paper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .generation import BOUNDARIES, PDE_FAMILIES
from .masks import MASK_PROTOCOL_NAMES
from .settings import SETTING_NAMES
from .splits import REGIMES, TIER_SIZES

TITLE = "PDE-OBS: A Controlled Partial-Observation Benchmark for PDE Dynamics"
CENTRAL_QUESTION = (
    "How should we evaluate PDE learning under partial observations in a controlled, "
    "factorized, and reproducible way?"
)

TASKS = (
    {
        "id": "T1",
        "name": "sparse_recovery",
        "status": "executable_field_task",
        "metrics": ["relative_l2", "mse", "spectral", "validated_pde_residual"],
    },
    {
        "id": "T2",
        "name": "forward_prediction",
        "status": "executable_field_task",
        "metrics": ["relative_l2", "spectral", "validated_pde_residual"],
    },
    {
        "id": "T3",
        "name": "inverse_prediction",
        "status": "executable_field_task",
        "metrics": ["relative_l2", "parameter_error"],
    },
    {
        "id": "T4",
        "name": "semantic_retrieval",
        "status": "lightweight_anchor_api",
        "metrics": ["recall_at_k", "map", "ndcg", "semantic_ambiguity"],
    },
    {
        "id": "T5",
        "name": "world_modeling",
        "status": "executable_field_task",
        "metrics": ["rollout_relative_l2", "spectral", "energy", "enstrophy", "stability"],
    },
    {
        "id": "T6",
        "name": "solver_routing",
        "status": "lightweight_anchor_api",
        "metrics": ["solver_accuracy", "solver_regret"],
    },
    {
        "id": "T7",
        "name": "foundation_transfer",
        "status": "lightweight_protocol_only",
        "metrics": ["linear_probe", "few_shot", "ood_transfer"],
    },
)

SPLITS = (
    "iid",
    "boundary_ood",
    "setting_ood",
    "parameter_ood",
    "combination_ood",
    "mask_ood",
    "time_horizon_ood",
)

MINIMUM_BASELINES = (
    "zero_fill",
    "nearest_interpolation",
    "rbf_interpolation",
    "unet",
    "fno",
    "cno",
    "continuous_latent_ann",
    "flat_vq",
    "rq_without_prefix_supervision",
    "persistence",
    "autoregressive_unet",
    "autoregressive_fno",
    "convlstm",
    "scratch",
    "mae_small",
    "supervised_multitask_small",
)

ANALYSES = (
    "observation_ratio_difficulty",
    "observation_pattern_difficulty",
    "pde_family_difficulty",
    "boundary_generalization",
    "setting_generalization",
    "physical_parameter_extrapolation",
    "combination_ood",
    "time_horizon_difficulty",
    "spectral_high_frequency_error",
    "semantic_ambiguity",
    "solver_routing_difficulty",
    "difficulty_heatmap",
    "data_scaling",
    "model_scaling",
    "failure_cases",
)

EXCLUDED_CLAIMS = (
    "best_semantic_id_method",
    "best_world_model",
    "new_foundation_model",
    "state_of_the_art_on_every_task",
    "paper_grade_ground_truth_before_numerical_validation",
)


def benchmark_contract() -> dict[str, Any]:
    """Return a detached JSON-safe representation of the frozen paper scope."""

    macro_cases = len(PDE_FAMILIES) * len(BOUNDARIES) * len(SETTING_NAMES)
    return deepcopy(
        {
            "schema_version": "pdeobs.benchmark-paper/v1",
            "title": TITLE,
            "central_question": CENTRAL_QUESTION,
            "contribution": [
                "dataset_design",
                "task_suite",
                "official_splits",
                "metrics",
                "anchor_leaderboard",
                "difficulty_analysis",
                "one_line_tools",
            ],
            "dataset": {
                "pde_families": list(PDE_FAMILIES),
                "boundaries": list(BOUNDARIES),
                "settings": list(SETTING_NAMES),
                "regimes": list(REGIMES),
                "macro_cases": macro_cases,
                "regime_nodes": macro_cases * len(REGIMES),
                "samples_per_macro_case_full": 2000,
                "full_samples": macro_cases * 2000,
                "resolution": [128, 128],
                "canonical_arrays": {
                    "condition": "[N,H,W,V_cond]",
                    "trajectory": "[N,T,H,W,V_state]",
                    "geometry": "[N,H,W,1]",
                },
                "static_T": 1,
                "temporal_T": 9,
                "navier_stokes_T": 9,
                "tiers": {
                    name: {
                        "samples_per_macro_case": size,
                        "total_samples": macro_cases * size,
                    }
                    for name, size in TIER_SIZES.items()
                },
            },
            "tasks": list(TASKS),
            "splits": list(SPLITS),
            "masks": list(MASK_PROTOCOL_NAMES),
            "minimum_baselines": list(MINIMUM_BASELINES),
            "analyses": list(ANALYSES),
            "excluded_claims": list(EXCLUDED_CLAIMS),
            "publication_gate": {
                "official_release_published": False,
                "bundled_solver_fidelity": "compact_reference",
                "required_before_paper_data": [
                    "convergence_study",
                    "residual_validation",
                    "trusted_solver_comparison",
                    "full_factor_matrix_validation",
                    "versioned_checksummed_release_manifest",
                ],
            },
        }
    )


def _sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(str(item) for item in value)


def validate_dataset_config(config: Mapping[str, Any]) -> list[str]:
    """Return every way a generation mapping drifts from the paper contract."""

    issues: list[str] = []
    expected_sequences = {
        "families": PDE_FAMILIES,
        "boundaries": BOUNDARIES,
        "settings": SETTING_NAMES,
        "regimes": REGIMES,
    }
    for key, expected in expected_sequences.items():
        actual = _sequence(config.get(key))
        if actual != tuple(expected):
            issues.append(f"{key} must equal {list(expected)!r}; found {list(actual)!r}")

    actual_tiers = config.get("tiers")
    if not isinstance(actual_tiers, Mapping):
        issues.append("tiers must be a mapping")
    else:
        try:
            normalized_tiers = {str(key): int(value) for key, value in actual_tiers.items()}
        except (TypeError, ValueError):
            normalized_tiers = {}
        if normalized_tiers != dict(TIER_SIZES):
            issues.append(f"tiers must equal {dict(TIER_SIZES)!r}; found {normalized_tiers!r}")

    try:
        samples_per_case = int(config.get("samples_per_case", 2000))
    except (TypeError, ValueError):
        samples_per_case = -1
    if samples_per_case != 2000:
        issues.append("samples_per_case must be 2000")
    resolution = config.get("resolution", 128)
    if resolution != 128 and resolution != (128, 128) and resolution != [128, 128]:
        issues.append("resolution must be 128 or [128, 128]")
    try:
        trajectory_steps = int(config.get("trajectory_steps", config.get("time_steps", 9)))
    except (TypeError, ValueError):
        trajectory_steps = -1
    if trajectory_steps != 9:
        issues.append("trajectory_steps must be 9")

    splits = config.get("splits", {})
    if not isinstance(splits, Mapping):
        issues.append("splits must be a mapping")
    else:
        for name, expected in (("train", 0.70), ("validation", 0.15), ("test", 0.15)):
            try:
                actual = float(splits.get(name))
            except (TypeError, ValueError):
                actual = float("nan")
            if actual != expected:
                issues.append(f"splits.{name} must be {expected}")
        allocation = splits.get("regime_allocation_full")
        expected_allocation = {"low": 667, "medium": 667, "high": 666}
        if not isinstance(allocation, Mapping):
            normalized_allocation: dict[str, int] = {}
        else:
            try:
                normalized_allocation = {str(key): int(value) for key, value in allocation.items()}
            except (TypeError, ValueError):
                normalized_allocation = {}
        if normalized_allocation != expected_allocation:
            issues.append(
                "splits.regime_allocation_full must equal "
                f"{expected_allocation!r}; found {normalized_allocation!r}"
            )

    observations = config.get("observations", {})
    if not isinstance(observations, Mapping):
        issues.append("observations must be a mapping")
    else:
        train = observations.get("train", {})
        train_protocol = str(train.get("protocol", "")) if isinstance(train, Mapping) else ""
        if train_protocol != "random":
            issues.append("observations.train.protocol must be random")
        try:
            train_count = int(train.get("count", -1)) if isinstance(train, Mapping) else -1
        except (TypeError, ValueError):
            train_count = -1
        if train_count != 500:
            issues.append("observations.train.count must be exactly 500 at 128x128")
        evaluation = observations.get("evaluation", ())
        if not isinstance(evaluation, Sequence):
            issues.append("observations.evaluation must list official mask views")
        else:
            rows = [row for row in evaluation if isinstance(row, Mapping)]
            random_ratios: list[float] = []
            for row in rows:
                if str(row.get("protocol")) != "random":
                    continue
                try:
                    random_ratios.append(float(row.get("ratio")))
                except (TypeError, ValueError):
                    random_ratios.append(float("nan"))
            expected_ratios = [0.01, 0.03, 0.05, 0.10]
            if sorted(random_ratios) != expected_ratios:
                issues.append(
                    "observations.evaluation random ratios must equal "
                    f"{expected_ratios!r}; found {sorted(random_ratios)!r}"
                )
            structured = {
                "regular_grid",
                "block_missing",
                "line_sensors",
                "boundary_sensors",
                "clustered",
            }
            protocols = {str(row.get("protocol")) for row in rows}
            if not structured.issubset(protocols):
                issues.append("observations.evaluation must cover every structured mask protocol")
    return issues


def protocol_report(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the contract plus optional dataset-config conformance results."""

    report = benchmark_contract()
    if config is not None:
        issues = validate_dataset_config(config)
        report["validation"] = {"valid": not issues, "issues": issues}
    return report
