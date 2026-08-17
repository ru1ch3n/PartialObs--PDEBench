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
        "metrics": ["relative_l2", "mse", "spectral"],
    },
    {
        "id": "T2",
        "name": "forward_prediction",
        "status": "executable_field_task",
        "metrics": ["relative_l2", "spectral"],
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

# The paper's core observation comparison is deliberately separate from the
# broader minimum-anchor list above.  Rows without an executable implementation
# remain in the machine-readable plan so the website and campaign tooling can
# distinguish planned integrations from code that is actually present.
OBSERVATION_COUNTS_128 = {
    "random_1pct": {"observed_count": 164, "observed_fraction": 0.010009765625},
    "random_3pct": {"observed_count": 500, "observed_fraction": 0.030517578125},
    "random_5pct": {"observed_count": 819, "observed_fraction": 0.04998779296875},
    "random_10pct": {"observed_count": 1638, "observed_fraction": 0.0999755859375},
    "regular_grid": {"observed_count": 441, "observed_fraction": 0.02691650390625},
    "block_missing": {"observed_count": 12288, "observed_fraction": 0.75},
    "line_sensors": {"observed_count": 508, "observed_fraction": 0.031005859375},
    "boundary_sensors": {"observed_count": 508, "observed_fraction": 0.031005859375},
    "clustered_sensors": {"observed_count": 492, "observed_fraction": 0.030029296875},
}

CORE_OBSERVATION_METHODS = (
    {
        "method_id": "rbf",
        "label": "RBF / interpolation",
        "execution_status": "executable_builtin",
        "registry_name": "rbf",
        "fit_scope": "none",
        "mask_specific_training": False,
        "note": "Transparent evaluation-only baseline; no training job.",
    },
    {
        "method_id": "gappy_pod",
        "label": "Gappy POD / PCA",
        "execution_status": "planning_only",
        "registry_name": None,
        "fit_scope": "once_per_pde_training_split",
        "mask_specific_training": False,
        "note": "Fit seven leakage-free PDE bases; no implementation is bundled yet.",
    },
    {
        "method_id": "unet",
        "label": "Mask-channel U-Net",
        "execution_status": "executable_compact_reference",
        "registry_name": "unet",
        "fit_scope": "once_per_pde_and_observation",
        "mask_specific_training": True,
        "note": "Bundled compact reference; not an exact paper reproduction.",
    },
    {
        "method_id": "fno",
        "label": "Mask-channel FNO",
        "execution_status": "executable_compact_reference",
        "registry_name": "fno",
        "fit_scope": "once_per_pde_and_observation",
        "mask_specific_training": True,
        "note": "Bundled compact reference; not an exact paper reproduction.",
    },
    {
        "method_id": "cno",
        "label": "CNO",
        "execution_status": "executable_compact_reference",
        "registry_name": "cno",
        "fit_scope": "once_per_pde_and_observation",
        "mask_specific_training": True,
        "note": "The bundled model is CNO-like, not an exact CNO reproduction.",
    },
    {
        "method_id": "deeponet",
        "label": "DeepONet",
        "execution_status": "planning_only",
        "registry_name": None,
        "fit_scope": "once_per_pde_and_observation",
        "mask_specific_training": True,
        "note": "Requires a versioned external adapter or future plugin.",
    },
    {
        "method_id": "pinn_or_pino",
        "label": "PINN or PINO",
        "execution_status": "planning_only_choice_required",
        "registry_name": None,
        "fit_scope": "once_per_pde_and_observation",
        "mask_specific_training": True,
        "implementation_choice_required": True,
        "note": "The job accounting assumes one amortized operator-level implementation.",
    },
    {
        "method_id": "transolver_or_gnot",
        "label": "Transolver or GNOT",
        "execution_status": "planning_only_choice_required",
        "registry_name": None,
        "fit_scope": "once_per_pde_and_observation",
        "mask_specific_training": True,
        "implementation_choice_required": True,
        "note": "Choose and freeze exactly one implementation before counting results.",
    },
    {
        "method_id": "diffusionpde",
        "label": "DiffusionPDE",
        "execution_status": "planning_only_external_adapter",
        "registry_name": None,
        "fit_scope": "once_per_pde_prior",
        "mask_specific_training": False,
        "note": "Use the exact upstream prior and condition it on all nine masks.",
    },
    {
        "method_id": "fundps",
        "label": "FunDPS",
        "execution_status": "planning_only_external_adapter",
        "registry_name": None,
        "fit_scope": "once_per_pde_prior",
        "mask_specific_training": False,
        "note": "Use the exact upstream prior and condition it on all nine masks.",
    },
)

NORMAL_OBSERVATION_TRAINING_METHODS = (
    "unet",
    "fno",
    "cno",
    "deeponet",
    "pinn_or_pino",
    "transolver_or_gnot",
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


def observation_training_contract() -> dict[str, Any]:
    """Return the frozen matched-mask campaign policy and planning arithmetic."""

    observations = list(MASK_PROTOCOL_NAMES)
    transfer_masks = [name for name in observations if name != "random_3pct"]
    return deepcopy(
        {
            "schema_version": "pdeobs.observation-training/v1",
            "task": "sparse_recovery",
            "primary": {
                "name": "matched_mask_iid",
                "split": "iid",
                "observation_protocols": observations,
                "normal_trainable_methods": list(NORMAL_OBSERVATION_TRAINING_METHODS),
                "training_mask_equals_evaluation_mask": True,
                "independent_checkpoint_per_pde_and_observation": True,
                "reuse_weights_across_observations": False,
                "reuse_immutable_physical_dataset": True,
            },
            "secondary": {
                "name": "mask_transfer",
                "split": "mask_ood",
                "training_mask": "random_3pct",
                "evaluation_masks": transfer_masks,
                "separate_result_table": True,
                "note": (
                    "This train-on-3%-and-transfer view is not a substitute for the "
                    "primary matched-mask comparison."
                ),
            },
            "observation_counts_128": {
                name: dict(OBSERVATION_COUNTS_128[name]) for name in observations
            },
            "methods": [dict(row) for row in CORE_OBSERVATION_METHODS],
            "dataset_accounting": {
                "pde_count": 7,
                "observation_count": 9,
                "method_row_count": 10,
                "full": {
                    "total_records": 560_000,
                    "records_per_pde": 80_000,
                    "training_records_per_pde": 56_000,
                },
                "medium": {
                    "total_records": 140_000,
                    "records_per_pde": 20_000,
                    "training_records_per_pde_approximately": 14_000,
                    "training_records_per_pde_range_for_frozen_seed": [13_916, 14_016],
                    "seed": 20_260_804,
                },
            },
            "campaign_accounting": {
                "scope": "one sparse-recovery task and one IID result table",
                "iid_result_cells": 630,
                "normal_training_jobs_one_seed": 378,
                "normal_training_jobs_pinn_or_pino_three_seeds": 504,
                "gappy_pod_fits": 7,
                "external_prior_jobs": {"compatible_pretrained": 0, "retrain_per_pde": 14},
                "total_jobs_one_seed": {"minimum": 385, "maximum": 399},
                "total_jobs_pinn_or_pino_three_seeds": {"minimum": 511, "maximum": 525},
                "diffusionpde_fundps_observation_cells": 126,
                "five_factor_split_result_cells": 3_150,
                "counts_assume_one_implementation_per_choice_row": True,
                "result_cells_are_not_scheduler_jobs": True,
            },
            "compute_planning": {
                "status": "unmeasured_planning_scenario",
                "hardware": "12 dedicated NVIDIA A6000 GPUs",
                "duration_days": 10,
                "theoretical_gpu_hours": 2_880,
                "usable_gpu_hours_at_75_to_80_percent": [2_160, 2_304],
                "attachment_rounded_safe_gpu_hours": [2_100, 2_300],
                "full_gpu_hours_pinn_or_pino_one_seed": [4_200, 4_600],
                "full_gpu_hours_pinn_or_pino_three_seeds": [7_000, 7_400],
                "medium_gpu_hours_pinn_or_pino_three_seeds": [1_800, 2_300],
                "estimated_wall_days_with_overhead": {
                    "full_one_seed": [17, 22],
                    "full_three_seeds": [28, 32],
                    "medium_three_seeds": [8, 10],
                },
                "pilot_required": True,
                "warning": (
                    "These are unmeasured planning estimates, not benchmark results. "
                    "Dataset-size quartering is not a validated scaling law for PINNs, "
                    "diffusion sampling, or fixed-cost prior training. SeaWulf uses a "
                    "shared A100 queue, so this dedicated-A6000 scenario is not a SeaWulf "
                    "capacity promise."
                ),
            },
            "scientific_caveats": [
                (
                    "All current Navier-Stokes validation routes store one-channel "
                    "vorticity, but periodic, rectangular, and obstacle cases use different "
                    "registered velocity-reconstruction and geometry operators; a "
                    "cross-boundary checkpoint must retain that topology metadata."
                ),
                (
                    "Gappy POD must fit only on the canonical training split; fitting the basis "
                    "on validation or test records is leakage."
                ),
                (
                    "The block-missing view observes 75% of the grid and is not density-matched "
                    "to the approximately 3% structured views."
                ),
                (
                    "Classical per-instance PINN and amortized PINO are not interchangeable; "
                    "the frozen job counts require one operator-level implementation choice."
                ),
                (
                    "Final uncertainty comparisons should use a consistent seed policy across "
                    "stochastic learned methods and record diffusion sampling seeds."
                ),
            ],
        }
    )


def benchmark_contract() -> dict[str, Any]:
    """Return a detached JSON-safe representation of the frozen paper scope."""

    macro_cases = len(PDE_FAMILIES) * len(BOUNDARIES) * len(SETTING_NAMES)
    return deepcopy(
        {
            "schema_version": "pdeobs.benchmark-paper/v2",
            "title": TITLE,
            "central_question": CENTRAL_QUESTION,
            "contribution": [
                "dataset_design",
                "task_suite",
                "official_splits",
                "metrics",
                "anchor_leaderboard",
                "difficulty_analysis",
                "dataset_quality_control",
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
            "observation_training": observation_training_contract(),
            "minimum_baselines": list(MINIMUM_BASELINES),
            "analyses": list(ANALYSES),
            "dataset_quality": {
                "schema_version": "1.0",
                "profiles": ["report", "strict", "publication"],
                "default_profile": "report",
                "pde_loss_coverage": list(PDE_FAMILIES),
                "outputs": [
                    "sample_metadata",
                    "shard_quality_summary",
                    "dataset_quality_json",
                    "dataset_quality_csv",
                ],
                "prediction_residual_separate": True,
            },
            "excluded_claims": list(EXCLUDED_CLAIMS),
            "publication_gate": {
                "official_release_published": False,
                "bundled_solver_fidelity": "compact_reference",
                "required_before_paper_data": [
                    "convergence_study",
                    "residual_validation",
                    "trusted_solver_comparison",
                    "full_factor_matrix_validation",
                    "family_boundary_resolution_threshold_calibration",
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
    quality = config.get("quality", {})
    if not isinstance(quality, Mapping):
        issues.append("quality must be a mapping")
    else:
        if quality.get("enabled") is not True:
            issues.append("quality.enabled must be true")
        if quality.get("profile") != "report":
            issues.append("quality.profile must be report in the canonical generation config")
        if quality.get("require_pde_loss") is not True:
            issues.append("quality.require_pde_loss must be true")
    return issues


def protocol_report(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the contract plus optional dataset-config conformance results."""

    report = benchmark_contract()
    if config is not None:
        issues = validate_dataset_config(config)
        report["validation"] = {"valid": not issues, "issues": issues}
    return report
