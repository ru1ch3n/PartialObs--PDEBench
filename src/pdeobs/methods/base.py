"""Method interfaces and discovery for PDE-OBS baselines.

The registry in this module is deliberately local to the method layer.  It does
not depend on the dataset registry and can therefore be used by third-party
packages without importing the data-generation stack.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MethodCapabilities:
    """Machine-readable description used by CLIs and benchmark runners."""

    tasks: frozenset[str] = field(default_factory=lambda: frozenset({"recovery"}))
    trainable: bool = False
    temporal: bool = False
    requires_mask: bool = False
    supports_multichannel: bool = True
    reference_only: bool = False
    notes: str = ""

    def supports(self, task: str) -> bool:
        return task.replace("-", "_").lower() in self.tasks


@runtime_checkable
class Method(Protocol):
    """Small protocol shared by NumPy baselines and PyTorch models."""

    name: str
    capabilities: MethodCapabilities

    def predict(self, observations: Any, mask: Any | None = None, **kwargs: Any) -> Any:
        """Return a reconstruction or rollout."""


Factory = Callable[..., Any]
METHOD_REGISTRY: dict[str, Factory] = {}
_PRIMARY_NAMES: set[str] = set()


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def register_method(
    name: str | None = None,
    *,
    aliases: Iterable[str] = (),
    replace: bool = False,
) -> Callable[[Factory], Factory]:
    """Register a class/factory while preserving a decorator-friendly API."""

    def decorator(factory: Factory) -> Factory:
        primary = _normalise_name(name or getattr(factory, "name", factory.__name__))
        keys = (primary, *(_normalise_name(alias) for alias in aliases))
        for key in keys:
            if key in METHOD_REGISTRY and METHOD_REGISTRY[key] is not factory and not replace:
                raise ValueError(f"Method name already registered: {key}")
            METHOD_REGISTRY[key] = factory
        _PRIMARY_NAMES.add(primary)
        return factory

    return decorator


def discover_methods(group: str = "pdeobs.methods") -> dict[str, Factory]:
    """Load external method entry points.

    An entry point may expose a class/factory or a zero-argument hook that
    registers several methods and returns ``None``. Broken optional plugins do
    not prevent built-in baselines from being used; their errors are reported
    by :func:`method_discovery_errors`.
    """

    errors: dict[str, str] = {}
    try:
        entries = metadata.entry_points()
        selected = (
            entries.select(group=group) if hasattr(entries, "select") else entries.get(group, ())
        )
    except Exception as exc:  # pragma: no cover - unusual Python installation
        _DISCOVERY_ERRORS[group] = {"entry_points": repr(exc)}
        return dict(METHOD_REGISTRY)

    for entry in selected:
        try:
            loaded = entry.load()
            key = _normalise_name(entry.name)
            # Conventionally a function literally named ``register`` is a
            # zero-argument plugin hook. Other callables are method factories.
            if (
                callable(loaded)
                and not isinstance(loaded, type)
                and getattr(loaded, "__name__", "") == "register"
            ):
                result = loaded()
                if isinstance(result, Mapping):
                    for plugin_name, plugin in result.items():
                        register_method(str(plugin_name))(plugin)
                elif result is not None:
                    register_method(key)(result)
            elif isinstance(loaded, type) or callable(loaded):
                METHOD_REGISTRY.setdefault(key, loaded)
                _PRIMARY_NAMES.add(key)
        except Exception as exc:  # optional third-party plugin
            errors[entry.name] = repr(exc)
    _DISCOVERY_ERRORS[group] = errors
    return dict(METHOD_REGISTRY)


_DISCOVERY_ERRORS: dict[str, dict[str, str]] = {}


def method_discovery_errors(group: str = "pdeobs.methods") -> Mapping[str, str]:
    return dict(_DISCOVERY_ERRORS.get(group, {}))


def available_methods(*, discover: bool = False, include_aliases: bool = False) -> tuple[str, ...]:
    if discover:
        discover_methods()
    return tuple(sorted(METHOD_REGISTRY if include_aliases else _PRIMARY_NAMES))


def create_method(name: str, /, **kwargs: Any) -> Any:
    """Instantiate a registered method by a stable, case-insensitive name."""

    key = _normalise_name(name)
    if key not in METHOD_REGISTRY:
        discover_methods()
    try:
        factory = METHOD_REGISTRY[key]
    except KeyError as exc:
        choices = ", ".join(available_methods()) or "<none>"
        raise KeyError(f"Unknown method {name!r}. Available methods: {choices}") from exc
    return factory(**kwargs)


def capabilities_for(method: str | Any) -> MethodCapabilities:
    candidate = create_method(method) if isinstance(method, str) else method
    capabilities = getattr(candidate, "capabilities", None)
    if not isinstance(capabilities, MethodCapabilities):
        raise TypeError(f"{candidate!r} does not expose MethodCapabilities")
    return capabilities
