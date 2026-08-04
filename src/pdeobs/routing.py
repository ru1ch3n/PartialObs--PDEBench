"""Simple solver-routing baselines and oracle-regret evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .retrieval import ContinuousANNRetriever


def _loss_matrix(losses: Any) -> np.ndarray:
    matrix = np.asarray(losses, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("losses must have shape [samples, solvers]")
    return matrix


def oracle_solver(losses: Any, solver_names: Sequence[str] | None = None) -> np.ndarray:
    """Return the minimum-loss solver index (or name) for each sample."""

    matrix = _loss_matrix(losses)
    indices = np.nanargmin(matrix, axis=1)
    return np.asarray(solver_names, dtype=object)[indices] if solver_names is not None else indices


def _solver_indices(chosen: Any, solver_names: Sequence[str] | None) -> np.ndarray:
    values = np.asarray(chosen)
    if solver_names is None or np.issubdtype(values.dtype, np.integer):
        return values.astype(int)
    lookup = {name: index for index, name in enumerate(solver_names)}
    try:
        return np.asarray([lookup[str(item)] for item in values], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unknown chosen solver: {exc.args[0]}") from exc


def solver_accuracy(chosen: Any, losses: Any, solver_names: Sequence[str] | None = None) -> float:
    matrix = _loss_matrix(losses)
    selected = _solver_indices(chosen, solver_names)
    if len(selected) != len(matrix):
        raise ValueError("chosen solver count differs from sample count")
    return float(np.mean(selected == np.nanargmin(matrix, axis=1)))


def solver_regret(
    chosen: Any,
    losses: Any,
    solver_names: Sequence[str] | None = None,
    *,
    relative: bool = False,
    epsilon: float = 1e-12,
) -> float:
    """Mean excess error relative to an oracle per-instance router."""

    matrix = _loss_matrix(losses)
    selected = _solver_indices(chosen, solver_names)
    row = np.arange(len(matrix))
    actual = matrix[row, selected]
    oracle = np.nanmin(matrix, axis=1)
    regret = actual - oracle
    if relative:
        regret = regret / np.maximum(np.abs(oracle), epsilon)
    return float(np.nanmean(regret))


def routing_metrics(
    chosen: Any, losses: Any, solver_names: Sequence[str] | None = None
) -> dict[str, float]:
    return {
        "solver_accuracy": solver_accuracy(chosen, losses, solver_names),
        "solver_regret": solver_regret(chosen, losses, solver_names),
        "relative_solver_regret": solver_regret(chosen, losses, solver_names, relative=True),
    }


@dataclass
class RandomRouter:
    seed: int = 0

    def fit(self, losses: Any, solver_names: Sequence[str] | None = None, **_: Any) -> RandomRouter:
        matrix = _loss_matrix(losses)
        self.n_solvers_ = matrix.shape[1]
        self.solver_names_ = tuple(solver_names) if solver_names is not None else None
        self.rng_ = np.random.default_rng(self.seed)
        return self

    def predict(self, samples: int | Sequence[Any]) -> np.ndarray:
        count = samples if isinstance(samples, int) else len(samples)
        indices = self.rng_.integers(0, self.n_solvers_, size=count)
        return (
            np.asarray(self.solver_names_, dtype=object)[indices]
            if self.solver_names_ is not None
            else indices
        )


class FamilyRouter:
    """Choose the training-set best average solver for each PDE family."""

    def __init__(self, family_field: str = "pde") -> None:
        self.family_field = family_field

    def fit(
        self,
        metadata: Sequence[Mapping[str, Any]],
        losses: Any,
        solver_names: Sequence[str] | None = None,
    ) -> FamilyRouter:
        matrix = _loss_matrix(losses)
        if len(metadata) != len(matrix):
            raise ValueError("metadata and losses must have equal length")
        self.solver_names_ = tuple(solver_names) if solver_names is not None else None
        groups: dict[Any, list[int]] = defaultdict(list)
        for index, item in enumerate(metadata):
            groups[item.get(self.family_field)].append(index)
        self.routes_ = {
            key: int(np.nanargmin(np.nanmean(matrix[indices], axis=0)))
            for key, indices in groups.items()
        }
        self.default_ = int(np.nanargmin(np.nanmean(matrix, axis=0)))
        return self

    def predict(self, metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> np.ndarray:
        rows = [metadata] if isinstance(metadata, Mapping) else metadata
        indices = np.asarray(
            [self.routes_.get(item.get(self.family_field), self.default_) for item in rows],
            dtype=int,
        )
        return (
            np.asarray(self.solver_names_, dtype=object)[indices]
            if self.solver_names_ is not None
            else indices
        )


class MetadataRouter:
    """Best average solver for an exact metadata tuple, with global fallback."""

    def __init__(self, fields: Sequence[str] = ("pde", "boundary", "setting", "regime")) -> None:
        self.fields = tuple(fields)

    def _key(self, item: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(item.get(field) for field in self.fields)

    def fit(
        self,
        metadata: Sequence[Mapping[str, Any]],
        losses: Any,
        solver_names: Sequence[str] | None = None,
    ) -> MetadataRouter:
        matrix = _loss_matrix(losses)
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index, item in enumerate(metadata):
            groups[self._key(item)].append(index)
        self.routes_ = {
            key: int(np.nanargmin(np.nanmean(matrix[rows], axis=0))) for key, rows in groups.items()
        }
        self.default_ = int(np.nanargmin(np.nanmean(matrix, axis=0)))
        self.solver_names_ = tuple(solver_names) if solver_names is not None else None
        return self

    def predict(self, metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> np.ndarray:
        rows = [metadata] if isinstance(metadata, Mapping) else metadata
        indices = np.asarray(
            [self.routes_.get(self._key(item), self.default_) for item in rows], dtype=int
        )
        return (
            np.asarray(self.solver_names_, dtype=object)[indices]
            if self.solver_names_ is not None
            else indices
        )


class ContinuousLatentRouter:
    """Route by k-nearest latent examples and majority oracle solver."""

    def __init__(self, k: int = 5, metric: str = "cosine") -> None:
        self.k, self.metric = k, metric

    def fit(
        self, embeddings: Any, losses: Any, solver_names: Sequence[str] | None = None
    ) -> ContinuousLatentRouter:
        matrix = _loss_matrix(losses)
        if len(embeddings) != len(matrix):
            raise ValueError("embeddings and losses must have equal length")
        self.index_ = ContinuousANNRetriever(self.metric).fit(embeddings)
        self.oracle_ = np.nanargmin(matrix, axis=1)
        self.solver_names_ = tuple(solver_names) if solver_names is not None else None
        return self

    def predict(self, embeddings: Any) -> np.ndarray:
        neighbors = self.index_.query(embeddings, k=self.k)
        selected = []
        for row in self.oracle_[neighbors]:
            counts = Counter(row.tolist())
            selected.append(min(counts, key=lambda key: (-counts[key], key)))
        indices = np.asarray(selected, dtype=int)
        return (
            np.asarray(self.solver_names_, dtype=object)[indices]
            if self.solver_names_ is not None
            else indices
        )


FamilyOnlyRouter = FamilyRouter
MetadataOracleRouter = MetadataRouter
LatentRouter = ContinuousLatentRouter
regret = solver_regret


ROUTERS = {
    "random": RandomRouter,
    "family": FamilyRouter,
    "family_only": FamilyRouter,
    "metadata": MetadataRouter,
    "metadata_oracle": MetadataRouter,
    "continuous_latent": ContinuousLatentRouter,
    "latent": ContinuousLatentRouter,
}


def create_router(name: str, **kwargs: Any) -> Any:
    key = name.lower().replace("-", "_")
    try:
        return ROUTERS[key](**kwargs)
    except KeyError as exc:
        raise KeyError(f"Unknown router {name!r}: {', '.join(sorted(ROUTERS))}") from exc
