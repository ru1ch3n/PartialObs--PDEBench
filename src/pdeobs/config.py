"""Configuration loading with environment expansion and command-line overrides."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a benchmark configuration is invalid."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_MAX_CONFIG_DEPTH = 100
_MAX_CONFIG_CONTAINERS = 100_000


def _validate_config_structure(value: Any) -> None:
    """Reject ambiguous mapping keys and unsafe recursive configuration trees."""

    stack: list[tuple[Any, str, int, bool]] = [(value, "config", 0, False)]
    active_containers: set[int] = set()
    visited_containers = 0

    while stack:
        current, path, depth, exiting = stack.pop()
        is_mapping = isinstance(current, Mapping)
        is_sequence = isinstance(current, (list, tuple))
        if not (is_mapping or is_sequence):
            continue

        identity = id(current)
        if exiting:
            active_containers.remove(identity)
            continue
        if identity in active_containers:
            raise ConfigError(f"Recursive configuration value at {path!r} is not supported")
        if depth > _MAX_CONFIG_DEPTH:
            raise ConfigError(
                f"Configuration nesting exceeds {_MAX_CONFIG_DEPTH} levels at {path!r}"
            )
        visited_containers += 1
        if visited_containers > _MAX_CONFIG_CONTAINERS:
            raise ConfigError("Configuration contains too many nested containers")

        active_containers.add(identity)
        stack.append((current, path, depth, True))
        if is_mapping:
            children: list[tuple[Any, str]] = []
            for key, item in current.items():
                if not isinstance(key, str):
                    raise ConfigError(
                        f"Configuration mapping keys must be strings; {path!r} contains "
                        f"{key!r} ({type(key).__name__})"
                    )
                children.append((item, f"{path}.{key}"))
        else:
            children = [(item, f"{path}[{index}]") for index, item in enumerate(current)]
        for item, child_path in reversed(children):
            stack.append((item, child_path, depth + 1, False))


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ConfigError(f"Environment variable {name!r} is required by the configuration")

    return _ENV_PATTERN.sub(replace, value)


def expand_environment(value: Any) -> Any:
    """Recursively expand ``${NAME}`` and ``${NAME:-default}`` strings."""

    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _load_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in seen:
        raise ConfigError(f"Configuration include cycle at {resolved}")
    if not resolved.is_file():
        raise ConfigError(f"Configuration file does not exist: {resolved}")
    seen.add(resolved)
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except RecursionError as exc:
        raise ConfigError(f"Configuration YAML is nested too deeply: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Top level of {resolved} must be a mapping")

    includes = payload.pop("include", [])
    if isinstance(includes, (str, Path)):
        includes = [includes]
    result: dict[str, Any] = {}
    for include in includes:
        include_path = Path(str(include))
        if not include_path.is_absolute():
            include_path = resolved.parent / include_path
        _merge(result, _load_yaml(include_path, seen))
    _merge(result, payload)
    seen.remove(resolved)
    return result


def apply_overrides(config: dict[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides, parsing values as YAML scalars/collections."""

    result = deepcopy(config)
    for override in overrides or ():
        if "=" not in override:
            raise ConfigError(f"Override must have KEY=VALUE form: {override!r}")
        dotted_key, raw = override.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ConfigError(f"Override has an empty key: {override!r}")
        try:
            value = yaml.safe_load(raw)
        except RecursionError as exc:
            raise ConfigError(f"Override YAML is nested too deeply: {dotted_key!r}") from exc
        target = result
        for key in keys[:-1]:
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise ConfigError(f"Cannot set {dotted_key!r}; {key!r} is not a mapping")
            target = child
        target[keys[-1]] = value
    _validate_config_structure(result)
    return result


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Load, compose, override, and environment-expand a YAML configuration."""

    config = _load_yaml(Path(path), set())
    config = apply_overrides(config, overrides)
    return expand_environment(config)


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable 16-character identifier for a resolved configuration."""

    _validate_config_structure(config)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    return destination
