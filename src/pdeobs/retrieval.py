"""Anchor semantic-retrieval baselines and ambiguity metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _topk(k: int, size: int) -> int:
    if k < 1:
        raise ValueError("k must be at least one")
    if size < 1:
        raise ValueError("retriever has not been fitted with any candidates")
    return min(k, size)


@dataclass
class RandomRetriever:
    """Uniform random retrieval without replacement (per query)."""

    seed: int = 0

    def fit(
        self, candidates: Sequence[Any], metadata: Sequence[Mapping[str, Any]] | None = None
    ) -> RandomRetriever:
        self.candidates_ = candidates
        self.metadata_ = metadata
        self.size_ = len(candidates)
        self.rng_ = np.random.default_rng(self.seed)
        return self

    def query(self, queries: Any, k: int = 10) -> np.ndarray:
        count = len(queries) if hasattr(queries, "__len__") else int(queries)
        k = _topk(k, self.size_)
        return np.stack([self.rng_.choice(self.size_, size=k, replace=False) for _ in range(count)])


class SymbolicMetadataRetriever:
    """Rank candidates by exact matches along the semantic metadata tree."""

    DEFAULT_FIELDS = ("pde", "boundary", "setting", "regime", "sample_id")

    def __init__(
        self, fields: Sequence[str] = DEFAULT_FIELDS, weights: Sequence[float] | None = None
    ) -> None:
        self.fields = tuple(fields)
        self.weights = np.asarray(
            weights if weights is not None else np.arange(len(fields), 0, -1), dtype=float
        )
        if len(self.weights) != len(self.fields):
            raise ValueError("weights must have one value for every metadata field")

    def fit(
        self, metadata: Sequence[Mapping[str, Any]], candidates: Sequence[Any] | None = None
    ) -> SymbolicMetadataRetriever:
        self.metadata_ = list(metadata)
        self.candidates_ = candidates if candidates is not None else np.arange(len(metadata))
        return self

    def query(
        self, query_metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any], k: int = 10
    ) -> np.ndarray:
        queries = [query_metadata] if isinstance(query_metadata, Mapping) else list(query_metadata)
        k = _topk(k, len(self.metadata_))
        ranked: list[np.ndarray] = []
        for query in queries:
            scores = np.zeros(len(self.metadata_), dtype=float)
            prefix_alive = np.ones(len(self.metadata_), dtype=bool)
            for field, weight in zip(self.fields, self.weights, strict=True):
                if field not in query:
                    continue
                matches = np.asarray(
                    [candidate.get(field) == query[field] for candidate in self.metadata_]
                )
                scores += weight * matches
                # A small prefix bonus favors consistent hierarchical matches.
                prefix_alive &= matches
                scores += 0.01 * weight * prefix_alive
            # Stable sorting gives deterministic dataset-order tie breaks.
            ranked.append(np.argsort(-scores, kind="stable")[:k])
        return np.stack(ranked)


class ContinuousANNRetriever:
    """Exact chunked nearest-neighbor reference with cosine/Euclidean metrics.

    Despite the ANN name used by the benchmark task, this dependency-free
    reference uses exact search. External methods can register FAISS, ScaNN, or
    another approximate index behind the same ``fit/query`` interface.
    """

    def __init__(self, metric: str = "cosine", batch_size: int = 1024) -> None:
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.metric, self.batch_size = metric, batch_size

    def fit(
        self, embeddings: Any, metadata: Sequence[Mapping[str, Any]] | None = None
    ) -> ContinuousANNRetriever:
        vectors = np.asarray(embeddings, dtype=np.float64)
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("embeddings must be a non-empty [N,D] array")
        self.embeddings_ = vectors
        self.metadata_ = metadata
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self.index_ = vectors / np.maximum(norms, 1e-12)
        else:
            self.index_ = vectors
        return self

    def query(self, query_embeddings: Any, k: int = 10) -> np.ndarray:
        queries = np.asarray(query_embeddings, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries[None]
        if queries.ndim != 2 or queries.shape[1] != self.embeddings_.shape[1]:
            raise ValueError("queries must have shape [Q,D] matching the fitted embeddings")
        k = _topk(k, len(self.index_))
        output = np.empty((len(queries), k), dtype=int)
        for start in range(0, len(queries), self.batch_size):
            batch = queries[start : start + self.batch_size]
            if self.metric == "cosine":
                batch = batch / np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), 1e-12)
                scores = batch @ self.index_.T
                candidate = np.argpartition(-scores, k - 1, axis=1)[:, :k]
                order = np.take_along_axis(scores, candidate, axis=1).argsort(axis=1)[:, ::-1]
            else:
                distances = (
                    np.sum(batch**2, axis=1, keepdims=True)
                    + np.sum(self.index_**2, axis=1)[None]
                    - 2.0 * batch @ self.index_.T
                )
                candidate = np.argpartition(distances, k - 1, axis=1)[:, :k]
                order = np.take_along_axis(distances, candidate, axis=1).argsort(axis=1)
            output[start : start + len(batch)] = np.take_along_axis(candidate, order, axis=1)
        return output


class _KMeansCodebook:
    """Small deterministic Lloyd quantizer used by the anchor VQ baselines."""

    def __init__(self, codes: int = 64, iterations: int = 25, seed: int = 0) -> None:
        if codes < 1 or iterations < 1:
            raise ValueError("codes and iterations must be positive")
        self.codes, self.iterations, self.seed = int(codes), int(iterations), int(seed)

    def fit(self, vectors: np.ndarray) -> _KMeansCodebook:
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or not len(values):
            raise ValueError("quantizer input must be a non-empty [N,D] array")
        count = min(self.codes, len(values))
        rng = np.random.default_rng(self.seed)
        centers = values[rng.choice(len(values), count, replace=False)].copy()
        for _ in range(self.iterations):
            labels = self.encode(values, centers=centers)
            updated = centers.copy()
            distances = np.sum((values - centers[labels]) ** 2, axis=1)
            for code in range(count):
                members = values[labels == code]
                if len(members):
                    updated[code] = np.mean(members, axis=0)
                else:
                    updated[code] = values[int(np.argmax(distances))]
            if np.allclose(updated, centers, rtol=1e-7, atol=1e-9):
                centers = updated
                break
            centers = updated
        self.centers_ = centers
        return self

    def encode(self, vectors: Any, *, centers: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float64)
        selected = self.centers_ if centers is None else centers
        distances = (
            np.sum(values**2, axis=1, keepdims=True)
            + np.sum(selected**2, axis=1)[None]
            - 2.0 * values @ selected.T
        )
        return np.argmin(distances, axis=1)


class FlatVQRetriever:
    """One-level discrete-code retrieval anchor with no semantic supervision."""

    def __init__(self, codes: int = 64, iterations: int = 25, seed: int = 0) -> None:
        self.codes, self.iterations, self.seed = codes, iterations, seed

    def fit(
        self, embeddings: Any, metadata: Sequence[Mapping[str, Any]] | None = None
    ) -> FlatVQRetriever:
        values = np.asarray(embeddings, dtype=np.float64)
        self.quantizer_ = _KMeansCodebook(self.codes, self.iterations, self.seed).fit(values)
        self.codes_ = self.quantizer_.encode(values)
        self.quantized_ = self.quantizer_.centers_[self.codes_]
        self.metadata_ = metadata
        return self

    def encode(self, embeddings: Any) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim == 1:
            values = values[None]
        return self.quantizer_.encode(values)

    def query(self, query_embeddings: Any, k: int = 10) -> np.ndarray:
        values = np.asarray(query_embeddings, dtype=np.float64)
        if values.ndim == 1:
            values = values[None]
        query_codes = self.encode(values)
        quantized = self.quantizer_.centers_[query_codes]
        distances = np.sum((quantized[:, None] - self.quantized_[None]) ** 2, axis=-1)
        k = _topk(k, len(self.codes_))
        return np.argsort(distances, axis=1, kind="stable")[:, :k]


class ResidualQuantizationRetriever:
    """Multi-level residual quantization without semantic prefix supervision."""

    def __init__(
        self,
        codes: int = 32,
        levels: int = 3,
        iterations: int = 25,
        seed: int = 0,
    ) -> None:
        if levels < 1:
            raise ValueError("levels must be positive")
        self.codes, self.levels, self.iterations, self.seed = codes, levels, iterations, seed

    def fit(
        self, embeddings: Any, metadata: Sequence[Mapping[str, Any]] | None = None
    ) -> ResidualQuantizationRetriever:
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or not len(values):
            raise ValueError("embeddings must be a non-empty [N,D] array")
        residual = values.copy()
        reconstructed = np.zeros_like(values)
        codebooks, code_columns = [], []
        for level in range(self.levels):
            codebook = _KMeansCodebook(self.codes, self.iterations, self.seed + level).fit(residual)
            codes = codebook.encode(residual)
            contribution = codebook.centers_[codes]
            reconstructed += contribution
            residual = values - reconstructed
            codebooks.append(codebook)
            code_columns.append(codes)
        self.codebooks_ = codebooks
        self.codes_ = np.stack(code_columns, axis=1)
        self.quantized_ = reconstructed
        self.metadata_ = metadata
        return self

    def encode(self, embeddings: Any) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim == 1:
            values = values[None]
        residual = values.copy()
        reconstructed = np.zeros_like(values)
        columns = []
        for codebook in self.codebooks_:
            codes = codebook.encode(residual)
            reconstructed += codebook.centers_[codes]
            residual = values - reconstructed
            columns.append(codes)
        return np.stack(columns, axis=1), reconstructed

    def query(self, query_embeddings: Any, k: int = 10) -> np.ndarray:
        _, quantized = self.encode(query_embeddings)
        distances = np.sum((quantized[:, None] - self.quantized_[None]) ** 2, axis=-1)
        k = _topk(k, len(self.codes_))
        return np.argsort(distances, axis=1, kind="stable")[:, :k]


def _relevant_sets(relevant: Any, n_queries: int) -> list[set[Any]]:
    if isinstance(relevant, np.ndarray):
        relevant = relevant.tolist()
    values = list(relevant)
    if n_queries == 1 and (not values or not isinstance(values[0], (list, tuple, set, np.ndarray))):
        return [set(values)]
    if len(values) != n_queries:
        raise ValueError("relevant items must provide one set/list/scalar per query")
    output = []
    for item in values:
        if isinstance(item, (list, tuple, set, np.ndarray)):
            output.append(set(item))
        else:
            output.append({item})
    return output


def recall_at_k(retrieved: Any, relevant: Any, *, k: int | None = None) -> float:
    """Macro recall: fraction of relevant candidates found in each top-k list."""

    ranking = np.asarray(retrieved)
    if ranking.ndim == 1:
        ranking = ranking[None]
    use_k = ranking.shape[1] if k is None else min(int(k), ranking.shape[1])
    if use_k < 1:
        raise ValueError("k must be at least one")
    relevant_sets = _relevant_sets(relevant, len(ranking))
    scores = []
    for row, expected in zip(ranking[:, :use_k], relevant_sets, strict=True):
        scores.append(len(set(row.tolist()) & expected) / max(len(expected), 1))
    return float(np.mean(scores))


def label_recall_at_k(retrieved_labels: Any, target_labels: Any, *, k: int | None = None) -> float:
    """Fraction of queries whose target semantic label appears in top-k."""

    ranking = np.asarray(retrieved_labels, dtype=object)
    if ranking.ndim == 1:
        ranking = ranking[None]
    targets = np.asarray(target_labels, dtype=object).reshape(-1)
    if len(targets) != len(ranking):
        raise ValueError("target label count differs from query count")
    use_k = ranking.shape[1] if k is None else min(int(k), ranking.shape[1])
    return float(
        np.mean([target in row[:use_k] for row, target in zip(ranking, targets, strict=True)])
    )


def _expand_semantic_strings(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    flat = array.reshape(-1).tolist()
    if flat and all(isinstance(item, str) and "/" in item for item in flat):
        split = [item.split("/") for item in flat]
        if len({len(item) for item in split}) == 1:
            return np.asarray(split, dtype=object).reshape(*array.shape, len(split[0]))
    return array


def prefix_accuracy(
    predicted_codes: Any,
    target_codes: Any,
    *,
    max_depth: int | None = None,
) -> dict[str, float]:
    """Top-1 hierarchical prefix accuracy at every semantic depth."""

    predicted = _expand_semantic_strings(predicted_codes)
    target = _expand_semantic_strings(target_codes)
    if predicted.ndim == 3:  # [query, rank, depth]
        predicted = predicted[:, 0]
    if predicted.ndim == 1:
        predicted = predicted[None]
    if target.ndim == 1:
        target = target[None]
    if predicted.shape[0] != target.shape[0]:
        raise ValueError("predicted and target code batches differ")
    depth = min(predicted.shape[-1], target.shape[-1], max_depth or target.shape[-1])
    return {
        f"prefix_accuracy@{level}": float(
            np.mean(np.all(predicted[:, :level] == target[:, :level], axis=1))
        )
        for level in range(1, depth + 1)
    }


def retrieval_entropy(labels: Any, *, base: float = 2.0, normalize: bool = False) -> float:
    """Mean Shannon entropy of labels in each query's retrieval list."""

    values = np.asarray(labels, dtype=object)
    if values.ndim == 1:
        values = values[None]
    entropies = []
    for row in values:
        counts = np.asarray(list(Counter(row.tolist()).values()), dtype=float)
        probabilities = counts / counts.sum()
        entropy = -float(np.sum(probabilities * (np.log(probabilities) / np.log(base))))
        if normalize and len(counts) > 1:
            entropy /= np.log(len(row)) / np.log(base)
        entropies.append(entropy)
    return float(np.mean(entropies))


def semantic_diversity(labels: Any, *, normalized: bool = True) -> float:
    values = np.asarray(labels, dtype=object)
    if values.ndim == 1:
        values = values[None]
    unique = np.asarray([len(set(row.tolist())) for row in values], dtype=float)
    if normalized:
        unique /= max(values.shape[1], 1)
    return float(np.mean(unique))


def collision_rate(codes: Any, semantic_labels: Any) -> float:
    """Fraction of code buckets that collide across semantic labels."""

    code_values = np.asarray(codes, dtype=object)
    labels = np.asarray(semantic_labels, dtype=object)
    if code_values.ndim == 1:
        keys = code_values.tolist()
    else:
        keys = [tuple(row) for row in code_values.tolist()]
    if len(keys) != len(labels):
        raise ValueError("codes and semantic_labels must have equal length")
    buckets: dict[Any, set[Any]] = defaultdict(set)
    for key, label in zip(keys, labels.tolist(), strict=True):
        buckets[key].add(tuple(label) if isinstance(label, list) else label)
    if not buckets:
        return 0.0
    return float(np.mean([len(bucket) > 1 for bucket in buckets.values()]))


semantic_entropy = retrieval_entropy
top_k_semantic_diversity = semantic_diversity
retrieval_collision_rate = collision_rate


def evaluate_retrieval(
    retrieved_indices: Any,
    relevant_indices: Any,
    *,
    candidate_codes: Any | None = None,
    target_codes: Any | None = None,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    ranking = np.asarray(retrieved_indices)
    if ranking.ndim == 1:
        ranking = ranking[None]
    result = {
        f"recall@{k}": recall_at_k(ranking, relevant_indices, k=k)
        for k in ks
        if k <= ranking.shape[1]
    }
    if candidate_codes is not None:
        codes = np.asarray(candidate_codes, dtype=object)[ranking]
        # Last semantic component is used for ambiguity unless callers compute
        # entropy on a different level explicitly.
        labels = codes[..., -1] if codes.ndim >= 3 else codes
        result["retrieval_entropy"] = retrieval_entropy(labels)
        result["semantic_diversity"] = semantic_diversity(labels)
        if target_codes is not None:
            result.update(prefix_accuracy(codes, target_codes))
    return result


RETRIEVERS = {
    "random": RandomRetriever,
    "symbolic": SymbolicMetadataRetriever,
    "continuous_ann": ContinuousANNRetriever,
    "ann": ContinuousANNRetriever,
    "flat_vq": FlatVQRetriever,
    "vq": FlatVQRetriever,
    "residual_quantization": ResidualQuantizationRetriever,
    "rq": ResidualQuantizationRetriever,
}


def create_retriever(name: str, **kwargs: Any) -> Any:
    key = name.lower().replace("-", "_")
    try:
        return RETRIEVERS[key](**kwargs)
    except KeyError as exc:
        raise KeyError(f"Unknown retriever {name!r}: {', '.join(sorted(RETRIEVERS))}") from exc
