"""Atheris target for PDE-OBS configuration overrides and expansion."""

from __future__ import annotations

import os
import sys

import atheris

with atheris.instrument_imports():
    import yaml

    from pdeobs.config import ConfigError, apply_overrides, config_hash, expand_environment


_BASE_CONFIG = {
    "data": {"root": "./data", "resolution": 128},
    "training": {"learning_rate": 0.001, "mixed_precision": False},
}


def test_one_input(data: bytes) -> None:
    """Exercise YAML values, dotted keys, recursive expansion, and stable hashing."""

    key_bytes, separator, value_bytes = data.partition(b"\x00")
    if not separator:
        key_bytes, separator, value_bytes = data.partition(b"=")
    if not separator:
        return
    try:
        dotted_key = key_bytes.decode("utf-8")
        yaml_value = value_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return
    override = f"{dotted_key}={yaml_value}"

    os.environ["PDEOBS_FUZZ_ROOT"] = "/tmp/pdeobs-fuzz"
    try:
        resolved = expand_environment(apply_overrides(_BASE_CONFIG, [override]))
    except (ConfigError, yaml.YAMLError, UnicodeError, ValueError):
        return

    first_hash = config_hash(resolved)
    repeated_hash = config_hash(expand_environment(apply_overrides(_BASE_CONFIG, [override])))
    if first_hash != repeated_hash:
        raise RuntimeError("the same override produced a non-deterministic configuration hash")


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
