from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest

from pdeobs.config import load_config
from pdeobs.generation import BOUNDARIES, PDE_FAMILIES
from pdeobs.presets import (
    build_benchmark_preset,
    build_experiment_preset,
    default_generation_config,
    normalize_split,
    normalize_task,
)
from pdeobs.settings import SETTING_NAMES
from pdeobs.splits import REGIMES, official_ood_labels


def test_generation_preset_freezes_the_factorized_release_design() -> None:
    config = default_generation_config("full")

    assert len(config["families"]) == 7
    assert len(config["boundaries"]) == 4
    assert len(config["settings"]) == 10
    assert len(config["regimes"]) == 3
    assert config["tiers"]["full"] == 2000
    assert len(config["families"]) * len(config["boundaries"]) * len(config["settings"]) == 280


def test_experiment_preset_is_strict_and_uses_paper_aliases(tmp_path: Path) -> None:
    config = build_experiment_preset(
        task="sparse_recovery",
        model="fno",
        data=tmp_path / "pdeobs_medium",
        split="iid",
        mask="random_3%",
    )

    assert normalize_task("sparse-recovery") == "recovery"
    assert config["task"] == "recovery"
    assert config["method"]["name"] == "fno"
    assert config["data"]["mask"]["protocol"] == "random_3pct"
    assert config["data"]["verify_shards"] is True
    assert config["data"]["allow_split_fallback"] is False


def test_protocol_only_tasks_do_not_pretend_to_use_field_trainer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lightweight benchmark-paper protocol"):
        build_experiment_preset(
            task="semantic_retrieval",
            model="continuous_ann",
            data=tmp_path,
        )


def test_benchmark_preset_embeds_wheel_safe_experiment_mapping() -> None:
    config = build_benchmark_preset("fno_sparse_recovery", tier="medium")

    experiment = config["experiments"][0]
    assert isinstance(experiment["config"], dict)
    assert experiment["config"]["data"]["root"].endswith("pdeobs_medium")


def test_checked_in_paper_matrix_has_leak_free_factor_configs() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "experiment"
    for view in ("boundary", "setting", "parameter", "combination"):
        config = load_config(root / f"recovery_fno_{view}_ood.yaml")
        assert config["data"]["ood_view"] == view
        assert config["evaluation"]["ood_views"] == [view]
        expected_pde = "navier_stokes" if view == "combination" else "poisson"
        assert config["data"]["filters"] == {"pde": expected_pde}

        memberships = []
        filters = config["data"]["filters"]
        for pde, boundary, setting, regime in product(
            PDE_FAMILIES, BOUNDARIES, SETTING_NAMES, REGIMES
        ):
            factors = {
                "pde": pde,
                "boundary": boundary,
                "setting": setting,
                "regime": regime,
            }
            if all(factors[key] == value for key, value in filters.items()):
                memberships.append(
                    official_ood_labels(
                        pde=pde,
                        boundary=boundary,
                        setting=setting,
                        regime=regime,
                    )[f"{view}_ood"]
                )
        assert any(memberships), f"{view} config has no OOD factor nodes"
        assert not all(memberships), f"{view} config has no training factor nodes"

    combination = load_config(root / "recovery_fno_combination_ood.yaml")
    assert combination["data"]["state_representation"] == "velocity"
    assert combination["method"]["kwargs"]["in_channels"] == 2
    assert combination["method"]["kwargs"]["out_channels"] == 2

    suite = load_config(root / "benchmark_paper_anchors.yaml")
    assert len(suite["experiments"]) == 15


def test_public_ood_split_aliases_build_shape_compatible_presets(tmp_path: Path) -> None:
    boundary = build_experiment_preset(
        task="sparse_recovery",
        model="fno",
        data=tmp_path,
        split="boundary_ood",
    )
    assert normalize_split("boundary-ood") == "boundary"
    assert boundary["data"]["ood_view"] == "boundary"
    assert "boundary" not in boundary["data"]["filters"]

    combination = build_experiment_preset(
        task="sparse_recovery",
        model="fno",
        data=tmp_path,
        split="combination_ood",
    )
    assert combination["data"]["filters"]["pde"] == "navier_stokes"
    assert combination["data"]["state_representation"] == "velocity"
    assert combination["method"]["kwargs"]["in_channels"] == 2

    with pytest.raises(ValueError, match="explicit benchmark YAML"):
        build_experiment_preset(
            task="sparse_recovery",
            model="fno",
            data=tmp_path,
            split="mask_ood",
        )
