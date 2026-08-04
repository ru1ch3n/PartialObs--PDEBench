"""Manifest-driven, checksum-verified dataset downloader."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_release_manifest(source: str | Path) -> tuple[dict[str, Any], str | None]:
    text_source = str(source)
    parsed = urllib.parse.urlparse(text_source)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(text_source) as response:  # noqa: S310 - explicit user URL
            payload = json.loads(response.read().decode("utf-8"))
        return payload, text_source.rsplit("/", 1)[0] + "/"
    path = Path(source).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8")), path.parent.as_uri() + "/"


def _safe_destination(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise DownloadError(f"Manifest path escapes destination root: {relative!r}")
    return destination


def _selected_files(manifest: dict[str, Any], tier: str) -> Iterable[dict[str, Any]]:
    known_tiers = manifest.get("tiers", ["tiny", "debug", "signal", "medium", "full"])
    if tier not in known_tiers:
        raise DownloadError(f"Unknown tier {tier!r}; expected one of {known_tiers}")
    for entry in manifest.get("files", []):
        entry_tiers = entry.get("tiers", [entry.get("tier", "full")])
        if tier in entry_tiers:
            yield entry


def download_release(
    manifest_source: str | Path,
    destination: str | Path,
    tier: str,
    force: bool = False,
) -> list[Path]:
    manifest, base_url = load_release_manifest(manifest_source)
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []

    for entry in _selected_files(manifest, tier):
        relative = str(entry["path"])
        expected = str(entry["sha256"]).lower()
        output = _safe_destination(root, relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not force and sha256_file(output) == expected:
            completed.append(output)
            continue

        url = entry.get("url")
        if not url:
            if not base_url:
                raise DownloadError(f"No URL for {relative}")
            url = urllib.parse.urljoin(base_url, relative.replace(os.sep, "/"))
        partial = output.with_name(output.name + ".partial")
        try:
            parsed = urllib.parse.urlparse(str(url))
            if parsed.scheme == "file":
                shutil.copyfile(urllib.request.url2pathname(parsed.path), partial)
            else:
                with urllib.request.urlopen(str(url)) as response, partial.open("wb") as target:  # noqa: S310
                    shutil.copyfileobj(response, target)
            actual = sha256_file(partial)
            if actual != expected:
                raise DownloadError(f"Checksum mismatch for {relative}: {actual} != {expected}")
            partial.replace(output)
        finally:
            if partial.exists():
                partial.unlink()
        completed.append(output)

    if not completed:
        raise DownloadError(f"Manifest contains no files for tier {tier!r}")
    (root / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completed
