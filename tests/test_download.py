from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pdeobs.download import DownloadError, download_release


def test_local_manifest_download_is_verified_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "release"
    source.mkdir()
    payload = b"small deterministic fixture\n"
    (source / "fixture.bin").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "name": "fixture-release",
        "tiers": ["tiny"],
        "files": [
            {
                "path": "fixture.bin",
                "tier": "tiny",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
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


def test_release_manifest_schema_is_enforced(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps({"tiers": ["tiny"], "files": []}), encoding="utf-8")
    with pytest.raises(DownloadError, match="schema_version"):
        download_release(manifest, tmp_path / "output", "tiny")
