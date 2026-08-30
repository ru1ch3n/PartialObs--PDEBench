# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from pdeobs.reports import aggregate_records, load_records, write_csv_report, write_json_report
from pdeobs.routing import (
    ContinuousLatentRouter,
    FamilyRouter,
    MetadataRouter,
    oracle_solver,
    routing_metrics,
)


def routing_fixture():
    metadata = [{"pde": "heat", "boundary": "p"}] * 2 + [{"pde": "burgers", "boundary": "w"}] * 2
    losses = np.array([[0.1, 0.5], [0.2, 0.4], [0.8, 0.2], [0.9, 0.1]])
    return metadata, losses


def test_routing_oracle_metrics():
    _, losses = routing_fixture()
    choice = oracle_solver(losses)
    assert routing_metrics(choice, losses)["solver_regret"] == 0.0
    assert routing_metrics(choice, losses)["solver_accuracy"] == 1.0


def test_metadata_and_family_routers():
    metadata, losses = routing_fixture()
    for router in (FamilyRouter(), MetadataRouter()):
        choice = router.fit(metadata, losses).predict(metadata)
        np.testing.assert_array_equal(choice, [0, 0, 1, 1])


def test_continuous_latent_router():
    _, losses = routing_fixture()
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    router = ContinuousLatentRouter(k=1).fit(vectors, losses)
    np.testing.assert_array_equal(router.predict(vectors), [0, 0, 1, 1])


def test_csv_json_report_round_trip_and_aggregation(tmp_path):
    records = [
        {"method": "unet", "task": "recovery", "split": "iid", "rel_l2": 0.2},
        {"method": "unet", "task": "recovery", "split": "iid", "rel_l2": 0.4},
    ]
    csv_path, json_path = tmp_path / "runs.csv", tmp_path / "runs.json"
    write_csv_report(records, csv_path)
    write_json_report(records, json_path)
    assert len(load_records([csv_path, json_path])) == 4
    aggregate = aggregate_records(records)
    assert aggregate[0]["rel_l2.mean"] == pytest.approx(0.3)
    assert aggregate[0]["rel_l2.std"] == pytest.approx(0.1)
