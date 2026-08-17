from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pdeobs.cli import main
from pdeobs.config import load_config
from pdeobs.methods import available_methods
from pdeobs.presets import default_generation_config
from pdeobs.protocol import (
    ANALYSES,
    CORE_OBSERVATION_METHODS,
    OBSERVATION_COUNTS_128,
    TASKS,
    benchmark_contract,
    observation_training_contract,
    validate_dataset_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_contract_freezes_cardinality_tasks_and_claim_limits() -> None:
    contract = benchmark_contract()

    assert contract["dataset"]["macro_cases"] == 280
    assert contract["dataset"]["regime_nodes"] == 840
    assert contract["dataset"]["full_samples"] == 560_000
    assert contract["dataset"]["tiers"]["tiny"]["total_samples"] == 1_400
    assert contract["dataset"]["tiers"]["medium"]["total_samples"] == 140_000
    assert len(TASKS) == 7
    assert len(ANALYSES) == 15
    assert "best_semantic_id_method" in contract["excluded_claims"]
    assert contract["publication_gate"]["official_release_published"] is False


def test_observation_training_contract_freezes_matched_mask_policy_and_counts() -> None:
    protocol = observation_training_contract()

    assert protocol["schema_version"] == "pdeobs.observation-training/v1"
    assert protocol["primary"]["training_mask_equals_evaluation_mask"] is True
    assert protocol["primary"]["reuse_immutable_physical_dataset"] is True
    assert protocol["secondary"]["training_mask"] == "random_3pct"
    assert len(protocol["primary"]["observation_protocols"]) == 9
    assert len(CORE_OBSERVATION_METHODS) == 10
    assert protocol["campaign_accounting"]["iid_result_cells"] == 630
    assert protocol["campaign_accounting"]["normal_training_jobs_one_seed"] == 378
    assert protocol["campaign_accounting"]["normal_training_jobs_pinn_or_pino_three_seeds"] == 504
    assert protocol["campaign_accounting"]["total_jobs_one_seed"] == {
        "minimum": 385,
        "maximum": 399,
    }
    assert protocol["campaign_accounting"]["total_jobs_pinn_or_pino_three_seeds"] == {
        "minimum": 511,
        "maximum": 525,
    }
    assert protocol["compute_planning"]["theoretical_gpu_hours"] == 2880
    assert protocol["compute_planning"]["usable_gpu_hours_at_75_to_80_percent"] == [
        2160,
        2304,
    ]


def test_observation_training_statuses_match_the_method_registry() -> None:
    protocol = observation_training_contract()
    registered = set(available_methods())
    executable_rows = [row for row in protocol["methods"] if row["registry_name"]]
    planning_rows = [row for row in protocol["methods"] if row["registry_name"] is None]

    assert {row["method_id"] for row in executable_rows} == {"rbf", "unet", "fno", "cno"}
    assert all(row["registry_name"] in registered for row in executable_rows)
    assert len(planning_rows) == 6
    assert all(str(row["execution_status"]).startswith("planning_only") for row in planning_rows)
    assert OBSERVATION_COUNTS_128["random_3pct"]["observed_count"] == 500
    assert OBSERVATION_COUNTS_128["block_missing"]["observed_fraction"] == 0.75


def test_checked_in_medium_campaign_is_planning_only_and_matches_contract() -> None:
    campaign = load_config(ROOT / "configs/campaign/core_observation_medium.yaml")
    protocol = observation_training_contract()

    assert campaign["status"] == "mixed_executable_and_planning"
    assert campaign["dataset"]["reuse_immutable_physical_dataset"] is True
    assert campaign["primary_observation_policy"]["training_mask_equals_evaluation_mask"] is True
    assert campaign["secondary_observation_policy"]["training_mask"] == "random_3pct"
    assert campaign["accounting"]["iid_result_cells"] == 630
    assert campaign["accounting"]["total_jobs_pinn_or_pino_three_seeds"] == {
        "minimum": 511,
        "maximum": 525,
    }
    assert [row["method_id"] for row in campaign["methods"]] == [
        row["method_id"] for row in protocol["methods"]
    ]


def test_code_and_yaml_generation_defaults_conform_to_contract() -> None:
    assert validate_dataset_config(default_generation_config("full")) == []
    assert validate_dataset_config(load_config(ROOT / "configs/dataset/default.yaml")) == []


def test_protocol_validator_reports_factor_and_tier_drift() -> None:
    config = deepcopy(default_generation_config("full"))
    config["families"] = config["families"][:-1]
    config["tiers"]["full"] = 1999

    issues = validate_dataset_config(config)

    assert any(issue.startswith("families must equal") for issue in issues)
    assert any(issue.startswith("tiers must equal") for issue in issues)


def test_protocol_validator_rejects_mask_and_regime_allocation_drift() -> None:
    config = deepcopy(default_generation_config("full"))
    config["splits"]["regime_allocation_full"] = {"low": 2000, "medium": 0, "high": 0}
    config["observations"]["train"]["protocol"] = "regular_grid"
    config["observations"]["evaluation"] = [
        row for row in config["observations"]["evaluation"] if row.get("protocol") != "random"
    ] + [{"protocol": "random", "ratio": 0.99}]

    issues = validate_dataset_config(config)

    assert any("regime_allocation_full" in issue for issue in issues)
    assert "observations.train.protocol must be random" in issues
    assert any("random ratios must equal" in issue for issue in issues)


def test_protocol_validator_requires_default_quality_reporting() -> None:
    config = deepcopy(default_generation_config("full"))
    config["quality"] = {"enabled": False, "profile": "strict", "require_pde_loss": False}

    issues = validate_dataset_config(config)

    assert "quality.enabled must be true" in issues
    assert "quality.profile must be report in the canonical generation config" in issues
    assert "quality.require_pde_loss must be true" in issues


def test_protocol_cli_check_and_json(capsys) -> None:
    assert main(["protocol", "--check"]) == 0
    assert "Protocol check: PASS" in capsys.readouterr().out

    assert main(["protocol", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"schema_version": "pdeobs.benchmark-paper/v2"' in output
    assert '"full_samples": 560000' in output
