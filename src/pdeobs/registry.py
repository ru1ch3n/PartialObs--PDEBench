# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Small, dependency-free registries used throughout :mod:`pdeobs`.

The project deliberately keeps discovery here instead of hard-coding a switch in
the CLI.  A future method or PDE can therefore be provided by another package via
one of the ``pdeobs.*`` Python entry-point groups.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from importlib import metadata
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class RegistryError(LookupError):
    """Raised for duplicate or unknown registry entries."""


def normalize_name(name: str) -> str:
    """Return the canonical spelling used by all registries."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("registry names must be non-empty strings")
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


class Registry(Generic[T]):
    """A named collection supporting decorators and entry-point discovery."""

    def __init__(self, name: str, *, entry_point_group: str | None = None) -> None:
        self.name = normalize_name(name)
        self.entry_point_group = entry_point_group
        self._objects: dict[str, T] = {}
        self._aliases: dict[str, str] = {}

    def _install(
        self,
        key: str,
        obj: T,
        *,
        aliases: tuple[str, ...] = (),
        replace: bool = False,
    ) -> T:
        canonical = normalize_name(key)
        occupied = canonical in self._objects or canonical in self._aliases
        if occupied and not replace:
            raise RegistryError(f"{key!r} is already registered in {self.name}")
        self._objects[canonical] = obj
        self._aliases.pop(canonical, None)
        for alias in aliases:
            normalized = normalize_name(alias)
            # Hyphen/space spellings often normalize to the canonical key and
            # need no separate alias entry.
            if normalized == canonical:
                continue
            if self._aliases.get(normalized) == canonical:
                continue
            alias_occupied = normalized in self._objects or normalized in self._aliases
            if alias_occupied and not replace:
                raise RegistryError(f"alias {alias!r} is already registered in {self.name}")
            self._aliases[normalized] = canonical
        return obj

    def register(
        self,
        name: str | T | None = None,
        obj: T | None = None,
        *,
        aliases: tuple[str, ...] = (),
        replace: bool = False,
    ) -> T | Callable[[T], T]:
        """Register directly or as ``@registry.register('name')``.

        ``@registry.register`` is also supported and uses ``__name__``.
        """

        if name is not None and not isinstance(name, str):
            if obj is not None:
                raise TypeError("pass either a decorated object or obj=, not both")
            target = name
            inferred = getattr(target, "__name__", target.__class__.__name__)
            return self._install(inferred, target, aliases=aliases, replace=replace)

        if obj is not None:
            key = name or getattr(obj, "__name__", obj.__class__.__name__)
            return self._install(key, obj, aliases=aliases, replace=replace)

        def decorator(target: T) -> T:
            key = name or getattr(target, "__name__", target.__class__.__name__)
            return self._install(key, target, aliases=aliases, replace=replace)

        return decorator

    def resolve_name(self, name: str) -> str:
        key = normalize_name(name)
        return self._aliases.get(key, key)

    def get(self, name: str) -> T:
        canonical = self.resolve_name(name)
        try:
            return self._objects[canonical]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<empty>"
            raise RegistryError(
                f"unknown {self.name} entry {name!r}; available: {available}"
            ) from exc

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Resolve an entry and call it with the supplied arguments."""

        factory = self.get(name)
        if not callable(factory):
            if args or kwargs:
                raise TypeError(f"registered object {name!r} is not callable")
            return factory
        return factory(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._objects))

    def aliases(self) -> Mapping[str, str]:
        return dict(self._aliases)

    def items(self) -> tuple[tuple[str, T], ...]:
        return tuple((key, self._objects[key]) for key in self.names())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.resolve_name(name) in self._objects

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def discover(
        self,
        group: str | None = None,
        *,
        replace: bool = False,
        on_error: str = "warn",
    ) -> tuple[str, ...]:
        """Load objects advertised through an importlib entry-point group.

        Loading is explicit so importing :mod:`pdeobs` never imports arbitrary
        third-party packages.  ``on_error`` may be ``'warn'``, ``'ignore'``, or
        ``'raise'``.
        """

        selected_group = group or self.entry_point_group
        if not selected_group:
            raise ValueError(f"no entry-point group configured for {self.name}")
        if on_error not in {"warn", "ignore", "raise"}:
            raise ValueError("on_error must be 'warn', 'ignore', or 'raise'")

        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            matches = entry_points.select(group=selected_group)
        else:  # pragma: no cover - compatibility with old importlib-metadata
            matches = entry_points.get(selected_group, ())

        loaded: list[str] = []
        for entry_point in matches:
            try:
                plugin = entry_point.load()
                is_hook = (
                    callable(plugin)
                    and not isinstance(plugin, type)
                    and getattr(plugin, "__name__", "") == "register"
                )
                result = plugin() if is_hook else plugin
                if isinstance(result, Mapping):
                    for plugin_name, plugin_object in result.items():
                        self._install(str(plugin_name), plugin_object, replace=replace)
                        loaded.append(normalize_name(str(plugin_name)))
                elif result is not None:
                    self._install(entry_point.name, result, replace=replace)
                    loaded.append(normalize_name(entry_point.name))
            except Exception as exc:  # third-party plugin errors need context
                if on_error == "raise":
                    raise
                if on_error == "warn":
                    warnings.warn(
                        f"failed loading {selected_group}:{entry_point.name}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return tuple(loaded)


PDE_REGISTRY: Registry[Any] = Registry("pdes", entry_point_group="pdeobs.pdes")
SETTING_REGISTRY: Registry[Any] = Registry("settings", entry_point_group="pdeobs.settings")
MASK_REGISTRY: Registry[Any] = Registry("masks", entry_point_group="pdeobs.masks")
METHOD_REGISTRY: Registry[Any] = Registry("methods", entry_point_group="pdeobs.methods")
METRIC_REGISTRY: Registry[Any] = Registry("metrics", entry_point_group="pdeobs.metrics")

# Short aliases are convenient in plugins and retain backwards compatibility
# with early internal prototypes.
PDES = PDE_REGISTRY
SETTINGS = SETTING_REGISTRY
MASKS = MASK_REGISTRY
METHODS = METHOD_REGISTRY
METRICS = METRIC_REGISTRY

REGISTRIES: Mapping[str, Registry[Any]] = {
    "pdes": PDE_REGISTRY,
    "settings": SETTING_REGISTRY,
    "masks": MASK_REGISTRY,
    "methods": METHOD_REGISTRY,
    "metrics": METRIC_REGISTRY,
}


def get_registry(name: str) -> Registry[Any]:
    """Resolve a registry by singular/plural component name."""

    key = normalize_name(name)
    if not key.endswith("s"):
        key += "s"
    try:
        return REGISTRIES[key]
    except KeyError as exc:
        raise RegistryError(
            f"unknown registry {name!r}; available: {', '.join(REGISTRIES)}"
        ) from exc


def discover_plugins(
    groups: Mapping[str, Registry[Any]] | None = None,
    *,
    on_error: str = "warn",
) -> dict[str, tuple[str, ...]]:
    """Discover all built-in extension groups or a caller-provided subset."""

    selected = groups or {f"pdeobs.{name}": registry for name, registry in REGISTRIES.items()}
    return {
        group: registry.discover(group, on_error=on_error) for group, registry in selected.items()
    }


register_pde = PDE_REGISTRY.register
register_setting = SETTING_REGISTRY.register
register_mask = MASK_REGISTRY.register
register_method = METHOD_REGISTRY.register
register_metric = METRIC_REGISTRY.register
