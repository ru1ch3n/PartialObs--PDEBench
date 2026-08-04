from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pdeobs.download import download_release


def test_local_manifest_download_is_verified_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "release"
    source.mkdir()
    payload = b"small deterministic fixture\n"
    (source / "fixture.bin").write_bytes(payload)
    manifest = {
        "tiers": ["tiny"],
        "files": [
            {
                "path": "fixture.bin",
                "tier": "tiny",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "downloaded"

    first = download_release(manifest_path, output, "tiny")
    second = download_release(manifest_path, output, "tiny")

    assert first == second == [output.resolve() / "fixture.bin"]
    assert first[0].read_bytes() == payload
