# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import ssl
from pathlib import Path

import pytest

from pdeobs.download import (
    DownloadError,
    _secure_ssl_context,
    download_release,
    load_release_manifest,
)


def test_remote_download_context_requires_certificates_and_tls12() -> None:
    context = _secure_ssl_context()
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2


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


def test_remote_manifest_rejects_insecure_http_before_network_access() -> None:
    with pytest.raises(DownloadError, match="must use HTTPS"):
        load_release_manifest("http://example.test/manifest.json")


def test_manifest_rejects_insecure_artifact_url(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "name": "insecure-release",
        "tiers": ["tiny"],
        "files": [
            {
                "path": "fixture.bin",
                "tier": "tiny",
                "url": "http://example.test/fixture.bin",
                "sha256": hashlib.sha256(b"fixture").hexdigest(),
                "size": 7,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DownloadError, match="must use HTTPS"):
        download_release(manifest_path, tmp_path / "output", "tiny")
