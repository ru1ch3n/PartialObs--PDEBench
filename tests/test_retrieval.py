# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
import numpy as np

from pdeobs.retrieval import (
    ContinuousANNRetriever,
    FlatVQRetriever,
    RandomRetriever,
    ResidualQuantizationRetriever,
    SymbolicMetadataRetriever,
    collision_rate,
    prefix_accuracy,
    recall_at_k,
    retrieval_entropy,
    semantic_diversity,
)


def test_continuous_ann_finds_self():
    vectors = np.eye(5)
    ranking = ContinuousANNRetriever().fit(vectors).query(vectors, k=2)
    np.testing.assert_array_equal(ranking[:, 0], np.arange(5))
    assert recall_at_k(ranking, np.arange(5), k=1) == 1.0


def test_unsupervised_vq_and_rq_anchors_are_deterministic():
    rng = np.random.default_rng(9)
    embeddings = np.concatenate(
        (rng.normal(-2.0, 0.1, size=(8, 4)), rng.normal(2.0, 0.1, size=(8, 4)))
    )
    for retriever in (
        FlatVQRetriever(codes=4, iterations=10, seed=3),
        ResidualQuantizationRetriever(codes=4, levels=2, iterations=10, seed=3),
    ):
        first = retriever.fit(embeddings).query(embeddings[:2], k=4)
        second = retriever.fit(embeddings).query(embeddings[:2], k=4)
        np.testing.assert_array_equal(first, second)
        assert first.shape == (2, 4)


def test_symbolic_metadata_prefers_full_match():
    metadata = [
        {"pde": "heat", "boundary": "periodic", "setting": "smooth"},
        {"pde": "heat", "boundary": "wall", "setting": "smooth"},
        {"pde": "burgers", "boundary": "periodic", "setting": "front"},
    ]
    query = {"pde": "heat", "boundary": "wall", "setting": "smooth"}
    ranking = (
        SymbolicMetadataRetriever(fields=("pde", "boundary", "setting"))
        .fit(metadata)
        .query(query, k=3)
    )
    assert ranking[0, 0] == 1


def test_random_retriever_is_seed_reproducible():
    a = RandomRetriever(seed=4).fit(range(10)).query(range(3), k=4)
    b = RandomRetriever(seed=4).fit(range(10)).query(range(3), k=4)
    np.testing.assert_array_equal(a, b)
    assert all(len(set(row)) == 4 for row in a)


def test_semantic_ambiguity_metrics():
    labels = np.array([["a", "a", "b", "b"], ["a", "b", "c", "d"]])
    assert retrieval_entropy(labels) > 0
    assert semantic_diversity(labels) == 0.75
    result = prefix_accuracy(
        [["heat", "wall", "rough"], ["burgers", "periodic", "front"]],
        [["heat", "wall", "smooth"], ["burgers", "wall", "front"]],
    )
    assert result["prefix_accuracy@1"] == 1.0
    assert result["prefix_accuracy@2"] == 0.5
    assert collision_rate([[1, 2], [1, 2], [2, 1]], ["a", "b", "c"]) == 0.5
