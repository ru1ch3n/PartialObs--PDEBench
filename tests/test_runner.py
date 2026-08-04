from __future__ import annotations

from pathlib import Path

import numpy as np

from pdeobs.runner import _dataset
from pdeobs.schema import Sample
from pdeobs.storage import AtomicHDF5ShardWriter


def test_runner_materializes_official_ood_membership(tmp_path: Path) -> None:
    shard = tmp_path / "ood.h5"
    with AtomicHDF5ShardWriter(shard, expected_count=2, spec={"test": True}) as writer:
        for index, is_ood in enumerate((False, True)):
            field = np.full((8, 8, 1), index + 1, dtype=np.float32)
            writer.append(
                Sample(
                    condition=field,
                    trajectory=field[None],
                    geometry=np.zeros_like(field),
                    metadata={
                        "sample_id": f"ood-{index}",
                        "split": "test",
                        "pde": "poisson",
                        "boundary_ood": is_ood,
                    },
                )
            )
    config = {
        "task": "recovery",
        "data": {
            "root": str(tmp_path),
            "glob": "*.h5",
            "ood_view": "boundary",
            "mask": {"protocol": "random", "count": 4},
        },
    }

    iid = _dataset(config, "test", ood_membership=False)
    ood = _dataset(config, "test", ood_membership=True)

    assert iid is not None and ood is not None
    assert iid.metadata[0]["sample_id"] == "ood-0"
    assert ood.metadata[0]["sample_id"] == "ood-1"
