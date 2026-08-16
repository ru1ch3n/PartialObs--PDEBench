from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pdeobs.cli import main
from pdeobs.config import load_config
from pdeobs.presets import default_generation_config
from pdeobs.protocol import ANALYSES, TASKS, benchmark_contract, validate_dataset_config

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


def test_protocol_cli_check_and_json(capsys) -> None:
    assert main(["protocol", "--check"]) == 0
    assert "Protocol check: PASS" in capsys.readouterr().out

    assert main(["protocol", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"schema_version": "pdeobs.benchmark-paper/v1"' in output
    assert '"full_samples": 560000' in output
