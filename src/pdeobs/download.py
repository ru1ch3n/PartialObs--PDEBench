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
from urllib.error import HTTPError, URLError


class DownloadError(RuntimeError):
    pass


DEFAULT_RELEASE_MANIFEST_URL = (
    "https://github.com/ru1ch3n/PartialObs--PDEBench/releases/latest/download/manifest.json"
)

_TIERS = ("tiny", "debug", "signal", "medium", "full")


def _require_secure_remote_url(url: str, *, label: str) -> urllib.parse.ParseResult:
    """Allow local files and authenticated HTTPS, but never insecure remote delivery."""

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme and parsed.scheme not in {"file", "https"}:
        raise DownloadError(
            f"{label} must use HTTPS (or file:// for a trusted local file), not {parsed.scheme!r}"
        )
    return parsed


def _require_https_response(response: Any, *, label: str) -> None:
    """Reject a transport that was downgraded by a redirect."""

    final_url = str(response.geturl())
    if urllib.parse.urlparse(final_url).scheme != "https":
        raise DownloadError(f"{label} redirected to a non-HTTPS URL: {final_url}")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_release_manifest(source: str | Path) -> tuple[dict[str, Any], str | None]:
    text_source = str(source)
    if isinstance(source, Path) or (
        len(text_source) >= 3
        and text_source[0].isalpha()
        and text_source[1] == ":"
        and text_source[2] in {"/", "\\"}
    ):
        parsed = urllib.parse.urlparse("")
    else:
        parsed = _require_secure_remote_url(text_source, label="Release manifest URL")
    if parsed.scheme == "https":
        try:
            with urllib.request.urlopen(text_source) as response:  # noqa: S310
                _require_https_response(response, label="Release manifest URL")
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            suffix = (
                " The default endpoint is intentionally publication-gated: no official "
                "tier is available until numerical validation is complete. Supply "
                "--manifest for a validated private/local release."
                if text_source == DEFAULT_RELEASE_MANIFEST_URL
                else ""
            )
            raise DownloadError(
                f"Could not load release manifest {text_source}: {exc}.{suffix}"
            ) from exc
        return payload, text_source.rsplit("/", 1)[0] + "/"
    path = (
        (
            Path(urllib.request.url2pathname(parsed.path))
            if parsed.scheme == "file"
            else Path(source)
        )
        .expanduser()
        .resolve()
    )
    if not path.is_file():
        raise DownloadError(f"Release manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path.parent.as_uri() + "/"


def validate_release_manifest(manifest: dict[str, Any]) -> None:
    """Enforce the checked-in release-manifest v1 contract without extra deps."""

    if manifest.get("schema_version") != 1:
        raise DownloadError("Release manifest schema_version must be 1")
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        raise DownloadError("Release manifest name must be a non-empty string")
    tiers = manifest.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise DownloadError("Release manifest tiers must be a non-empty list")
    if len(set(tiers)) != len(tiers) or set(tiers) - set(_TIERS):
        raise DownloadError(f"Release manifest tiers must be unique values from {_TIERS}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise DownloadError("Release manifest files must be a list")
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise DownloadError(f"Release manifest files[{index}] must be an object")
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not path:
            raise DownloadError(f"Release manifest files[{index}].path is required")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise DownloadError(
                f"Release manifest files[{index}].sha256 must be 64 hexadecimal characters"
            )
        if "size" in entry and (not isinstance(entry["size"], int) or entry["size"] < 0):
            raise DownloadError(f"Release manifest files[{index}].size must be non-negative")
        entry_tiers = entry.get("tiers", [entry.get("tier", "full")])
        if not isinstance(entry_tiers, list):
            entry_tiers = [entry_tiers]
        if set(entry_tiers) - set(tiers):
            raise DownloadError(
                f"Release manifest files[{index}] references undeclared tiers {entry_tiers}"
            )


def _safe_destination(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise DownloadError(f"Manifest path escapes destination root: {relative!r}")
    return destination


def _selected_files(manifest: dict[str, Any], tier: str) -> Iterable[dict[str, Any]]:
    known_tiers = manifest["tiers"]
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
    validate_release_manifest(manifest)
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
        parsed = _require_secure_remote_url(str(url), label=f"Download URL for {relative}")
        offset = partial.stat().st_size if partial.exists() else 0
        if parsed.scheme == "file":
            source_path = Path(urllib.request.url2pathname(parsed.path))
            with source_path.open("rb") as source_handle:
                source_handle.seek(offset)
                with partial.open("ab" if offset else "wb") as target:
                    shutil.copyfileobj(source_handle, target)
        else:
            request = urllib.request.Request(  # noqa: S310 - manifest-controlled URL
                str(url), headers={"Range": f"bytes={offset}-"} if offset else {}
            )
            try:
                response = urllib.request.urlopen(request)  # noqa: S310
            except (HTTPError, URLError) as exc:
                raise DownloadError(
                    f"Download interrupted for {relative}; resumable partial retained at {partial}: {exc}"
                ) from exc
            with response:
                _require_https_response(response, label=f"Download URL for {relative}")
                append = bool(offset and getattr(response, "status", None) == 206)
                with partial.open("ab" if append else "wb") as target:
                    shutil.copyfileobj(response, target)
        expected_size = entry.get("size")
        if expected_size is not None and partial.stat().st_size != int(expected_size):
            raise DownloadError(
                f"Size mismatch for {relative}: {partial.stat().st_size} != {expected_size}; "
                f"partial retained at {partial}"
            )
        actual = sha256_file(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise DownloadError(f"Checksum mismatch for {relative}: {actual} != {expected}")
        partial.replace(output)
        completed.append(output)

    if not completed:
        raise DownloadError(f"Manifest contains no files for tier {tier!r}")
    (root / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completed
