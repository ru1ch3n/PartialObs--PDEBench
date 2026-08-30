# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Deterministic, NumPy-only reference generators for the seven PDE families.

The public entry point is :func:`generate_sample`.  Each result follows the
canonical unbatched layout and can be stacked by the dataset writer::

    condition   [H, W, V_cond]
    trajectory  [T, H, W, V_state]
    geometry    [H, W, 1]

Static families use ``T=1``.  Temporal families default to ``T=9``.  A small
runtime registry permits future family generators to be added without changing
call sites or importing the higher-level dataset schema.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..registry import PDE_REGISTRY
from . import burgers, darcy, heat, helmholtz, navier_stokes, poisson, reaction_diffusion
from .common import (
    BOUNDARY_ALIASES,
    FAMILY_ALIASES,
    REGIME_ALIASES,
    SETTING_ALIASES,
    SETTING_NAMES,
    PDEOutput,
    Resolution,
    normalize_boundary,
    normalize_family,
    normalize_regime,
    normalize_setting,
    parse_resolution,
)


@runtime_checkable
class PDEGenerator(Protocol):
    """Structural interface implemented by built-in and extension generators."""

    def __call__(
        self,
        boundary: str = "dirichlet",
        setting: str = "smooth_grf",
        regime: str = "medium",
        seed: int = 0,
        resolution: Resolution = 32,
        time_steps: int | None = None,
        **options: Any,
    ) -> PDEOutput: ...


Generator = Callable[..., PDEOutput]


def _registry_token(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("family name must be a non-empty string")
    text = value.strip().lower()
    for dash in ("\N{EN DASH}", "\N{EM DASH}", "\N{MINUS SIGN}"):
        text = text.replace(dash, "-")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


FAMILY_GENERATORS: dict[str, Generator] = {
    "darcy": darcy.generate,
    "poisson": poisson.generate,
    "helmholtz": helmholtz.generate,
    "heat": heat.generate,
    "reaction_diffusion": reaction_diffusion.generate,
    "burgers": burgers.generate,
    "navier_stokes": navier_stokes.generate,
}
BUILTIN_FAMILY_GENERATORS = dict(FAMILY_GENERATORS)
PDE_GENERATORS = FAMILY_GENERATORS
FAMILY_NAMES = tuple(FAMILY_GENERATORS)
PDE_FAMILIES = FAMILY_NAMES
STATIC_FAMILIES = frozenset(("darcy", "poisson", "helmholtz"))
TEMPORAL_FAMILIES = frozenset(("heat", "reaction_diffusion", "burgers", "navier_stokes"))
BOUNDARY_NAMES = ("dirichlet", "neumann", "periodic", "robin")
REGIME_NAMES = ("low", "medium", "high")
_GENERATOR_ALIASES: dict[str, str] = dict(FAMILY_ALIASES)
_PLUGINS_DISCOVERED = False

generate_darcy = darcy.generate
generate_poisson = poisson.generate
generate_helmholtz = helmholtz.generate
generate_heat = heat.generate
generate_reaction_diffusion = reaction_diffusion.generate
generate_burgers = burgers.generate
generate_navier_stokes = navier_stokes.generate

_BUILTIN_ALIASES: dict[str, tuple[str, ...]] = {
    "darcy": ("f0", "darcy_flow", "darcy-flow"),
    "poisson": ("f1",),
    "helmholtz": ("f2",),
    "heat": ("f3", "diffusion", "heat_diffusion", "heat-diffusion"),
    "reaction_diffusion": ("f4", "reaction-diffusion", "reactiondiffusion"),
    "burgers": ("f5", "burger"),
    "navier_stokes": ("f6", "ns", "navier-stokes", "navierstokes"),
}

# Install built-ins in the project-wide plugin registry while remaining safe
# when a test runner reloads this module.
for _family_name, _family_generator in FAMILY_GENERATORS.items():
    if _family_name not in PDE_REGISTRY:
        PDE_REGISTRY.register(
            _family_name,
            _family_generator,
            aliases=_BUILTIN_ALIASES[_family_name],
        )


def register_generator(
    name: str,
    generator: Generator,
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> None:
    """Register an extension generator under a canonical name and aliases."""

    canonical = _registry_token(name)
    if not callable(generator):
        raise TypeError("generator must be callable")
    if canonical in FAMILY_GENERATORS and not replace:
        raise ValueError(f"generator {canonical!r} is already registered")
    normalized_aliases = tuple(_registry_token(alias) for alias in aliases)
    for alias, token in zip(aliases, normalized_aliases, strict=True):
        existing = _GENERATOR_ALIASES.get(token)
        if existing is not None and existing != canonical and not replace:
            raise ValueError(f"generator alias {alias!r} is already registered for {existing!r}")

    PDE_REGISTRY.register(
        canonical,
        generator,
        aliases=aliases,
        replace=replace,
    )
    FAMILY_GENERATORS[canonical] = generator
    _GENERATOR_ALIASES[canonical] = canonical
    for token in normalized_aliases:
        _GENERATOR_ALIASES[token] = canonical


def discover_generators(*, on_error: str = "warn") -> tuple[str, ...]:
    """Load installed PDE entry points once for this Python process.

    Generation performs this check even for built-in family names so a plugin
    that explicitly registers a validated replacement cannot be bypassed merely
    because it is the first PDE operation in a fresh worker process.
    """

    global _PLUGINS_DISCOVERED
    if not _PLUGINS_DISCOVERED:
        PDE_REGISTRY.discover(on_error=on_error)
        _PLUGINS_DISCOVERED = True
    return available_families()


def get_generator(family: str) -> Generator:
    """Resolve a built-in ID/name or a runtime-registered family generator."""

    discover_generators(on_error="warn")
    token = _registry_token(family)
    canonical = _GENERATOR_ALIASES.get(token, token)
    try:
        return FAMILY_GENERATORS[canonical]
    except KeyError as exc:
        if token in PDE_REGISTRY:
            return PDE_REGISTRY.get(token)
        choices = ", ".join(sorted(set(FAMILY_GENERATORS) | set(PDE_REGISTRY.names())))
        raise ValueError(f"unknown PDE family {family!r}; choose one of: {choices}") from exc


def available_families() -> tuple[str, ...]:
    """Return registered canonical family names in deterministic order."""

    return tuple(dict.fromkeys((*FAMILY_GENERATORS, *PDE_REGISTRY.names())))


def generate_sample(
    family: str,
    boundary: str = "dirichlet",
    setting: str = "smooth_grf",
    regime: str = "medium",
    seed: int = 0,
    resolution: Resolution = 32,
    time_steps: int | None = None,
    **options: Any,
) -> PDEOutput:
    """Generate one deterministic PDE instance through the common API."""

    generator = get_generator(family)
    return generator(
        boundary=boundary,
        setting=setting,
        regime=regime,
        seed=seed,
        resolution=resolution,
        time_steps=time_steps,
        **options,
    )


generate = generate_sample


__all__ = [
    "BOUNDARY_ALIASES",
    "BOUNDARY_NAMES",
    "BUILTIN_FAMILY_GENERATORS",
    "FAMILY_ALIASES",
    "FAMILY_GENERATORS",
    "FAMILY_NAMES",
    "Generator",
    "PDE_FAMILIES",
    "PDE_GENERATORS",
    "PDEGenerator",
    "PDEOutput",
    "REGIME_ALIASES",
    "REGIME_NAMES",
    "Resolution",
    "SETTING_ALIASES",
    "SETTING_NAMES",
    "STATIC_FAMILIES",
    "TEMPORAL_FAMILIES",
    "available_families",
    "discover_generators",
    "generate",
    "generate_burgers",
    "generate_darcy",
    "generate_heat",
    "generate_helmholtz",
    "generate_navier_stokes",
    "generate_poisson",
    "generate_reaction_diffusion",
    "generate_sample",
    "get_generator",
    "normalize_boundary",
    "normalize_family",
    "normalize_regime",
    "normalize_setting",
    "parse_resolution",
    "register_generator",
]
