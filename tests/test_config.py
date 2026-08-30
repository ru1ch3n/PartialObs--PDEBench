# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from pdeobs.config import ConfigError, apply_overrides, config_hash, load_config


def test_config_include_override_and_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PDEOBS_TEST_ROOT", str(tmp_path / "data"))
    (tmp_path / "base.yaml").write_text(
        "seed: 1\noutput:\n  root: ${PDEOBS_TEST_ROOT}\nmodel:\n  width: 8\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "include: base.yaml\nmodel:\n  depth: 2\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path / "child.yaml", ["model.width=16", "flag=true"])
    assert config["output"]["root"] == str(tmp_path / "data")
    assert config["model"] == {"width": 16, "depth": 2}
    assert config["flag"] is True
    assert config_hash(config) == config_hash(dict(reversed(list(config.items()))))


def test_config_rejects_non_string_nested_mapping_keys() -> None:
    with pytest.raises(ConfigError, match="mapping keys must be strings"):
        apply_overrides({}, ["data.mask={1: 0, kind: random_ratio}"])

    with pytest.raises(ConfigError, match="mapping keys must be strings"):
        config_hash({"data": {1: "ambiguous"}})


def test_config_rejects_recursive_values() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    with pytest.raises(ConfigError, match="Recursive configuration value"):
        config_hash({"recursive": recursive})

    with pytest.raises(ConfigError, match="Recursive configuration value"):
        apply_overrides({}, ["recursive=&anchor [*anchor]"])


def test_checked_in_experiments_select_one_data_tier(tmp_path: Path, monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data"
    monkeypatch.setenv("PDEOBS_DATA", str(data_root))

    recovery = load_config(repository / "configs/experiment/recovery_unet.yaml")
    rollout = load_config(repository / "configs/experiment/rollout_fno.yaml")
    smoke = load_config(repository / "configs/experiment/recovery_unet_smoke.yaml")

    assert Path(recovery["data"]["root"]) == data_root / "signal"
    assert Path(rollout["data"]["root"]) == data_root / "signal"
    assert Path(smoke["data"]["root"]) == data_root / "smoke"
    assert smoke["data"]["mask"]["count"] == 8
    assert smoke["training"]["epochs"] == 1
