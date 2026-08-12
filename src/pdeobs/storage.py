"""Atomic, resumable HDF5 shards and a process-safe lazy reader.

Each generation task owns one shard.  Rows are first appended to ``*.partial``;
only a validated shard is atomically renamed to its public ``.h5`` name.  A
per-shard manifest and SHA-256 make resubmission safe without concurrent writes
to a shared catalogue.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .schema import SCHEMA_VERSION, Sample, json_safe

try:  # imported lazily enough that light CLI commands work without HDF5 extras
    import h5py  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on optional environment
    h5py = None


class StorageError(RuntimeError):
    """Base class for invalid or incomplete shards."""


class IncompleteShardError(StorageError):
    pass


def _canonical_json(value: Any) -> str:
    """Serialize metadata for exact, order-independent integrity comparisons."""

    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


_NON_CONTENT_SPEC_FIELDS = frozenset({"job_id", "output_path"})


def _content_provenance(value: Any) -> Any:
    """Remove volatile execution context but retain generator identity."""

    if not isinstance(value, Mapping):
        return value
    stable = {str(key): item for key, item in value.items()}
    for key in ("captured_at", "captured_at_utc", "slurm"):
        stable.pop(key, None)
    runtime = stable.get("runtime")
    if isinstance(runtime, Mapping):
        stable_runtime = {str(key): item for key, item in runtime.items()}
        for key in ("cwd", "executable", "hostname", "path", "working_directory"):
            stable_runtime.pop(key, None)
        if stable_runtime:
            stable["runtime"] = stable_runtime
        else:
            stable.pop("runtime", None)
    return stable


def _content_generation_spec(value: Any) -> Any:
    """Return the stable, content-defining part of a generation job spec.

    Full provenance remains stored in the HDF5 attributes and completion
    manifest, but volatile audit data must not make an otherwise identical
    generation rerun look like a different shard.  These fields are all
    top-level attributes of ``GenerationJob.to_dict``; nested solver options
    remain part of the content identity.
    """

    safe = json_safe(value)
    if not isinstance(safe, Mapping):
        return safe
    stable: dict[str, Any] = {}
    for key, item in safe.items():
        name = str(key)
        if name in _NON_CONTENT_SPEC_FIELDS:
            continue
        if name == "provenance":
            item = _content_provenance(item)
            if not item:
                continue
        stable[name] = item
    return stable


def generation_specs_match(left: Any, right: Any) -> bool:
    """Compare the stable content identity of two generation job specs."""

    return _canonical_json(_content_generation_spec(left)) == _canonical_json(
        _content_generation_spec(right)
    )


def _specs_match(left: Any, right: Any) -> bool:
    return generation_specs_match(left, right)


def _full_specs_match(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _require_h5py() -> Any:
    if h5py is None:
        raise ModuleNotFoundError(
            "HDF5 support requires h5py; install the pdeobs data dependencies"
        )
    return h5py


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    _atomic_text(
        Path(path),
        json.dumps(json_safe(data), sort_keys=True, indent=indent) + "\n",
    )


def write_jsonl_manifest(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Atomically write an array-job or shard catalogue as JSON Lines."""

    destination = Path(path)
    text = "".join(
        json.dumps(json_safe(dict(row)), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_text(destination, text)
    return destination


def read_jsonl_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise StorageError(f"invalid JSON on {path}:{line_number}") from exc
    return rows


def shard_sidecars(path: str | Path) -> dict[str, Path]:
    shard = Path(path)
    return {
        "manifest": shard.with_suffix(".manifest.json"),
        "checksum": shard.with_suffix(".sha256"),
        "metadata_csv": shard.with_suffix(".metadata.csv"),
        "metadata_json": shard.with_suffix(".metadata.json"),
    }


def read_shard_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = shard_sidecars(path)["manifest"]
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise IncompleteShardError(f"missing or invalid manifest: {manifest_path}") from exc


def is_shard_complete(
    path: str | Path,
    *,
    expected_count: int | None = None,
    verify_checksum: bool = True,
) -> bool:
    shard = Path(path)
    if not shard.is_file():
        return False
    try:
        manifest = read_shard_manifest(shard)
    except IncompleteShardError:
        return False
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False
    if expected_count is not None and int(manifest.get("sample_count", -1)) != int(expected_count):
        return False
    if int(manifest.get("bytes", -1)) != shard.stat().st_size:
        return False
    if verify_checksum and manifest.get("sha256") != sha256_file(shard):
        return False
    return True


def _flatten_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        safe = json_safe(value)
        if isinstance(safe, (dict, list)):
            flattened[str(key)] = json.dumps(safe, sort_keys=True, separators=(",", ":"))
        else:
            flattened[str(key)] = safe
    return flattened


class AtomicHDF5ShardWriter:
    """Append samples to a recoverable partial file and atomically commit it."""

    DATASET_NAMES = ("condition", "trajectory", "geometry", "metadata")

    def __init__(
        self,
        path: str | Path,
        *,
        expected_count: int | None = None,
        spec: Mapping[str, Any] | None = None,
        resume: bool = True,
        overwrite: bool = False,
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
    ) -> None:
        module = _require_h5py()
        self.path = Path(path)
        if self.path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("HDF5 shard paths must end in .h5 or .hdf5")
        if expected_count is not None and int(expected_count) < 0:
            raise ValueError("expected_count must be non-negative")
        self.expected_count = None if expected_count is None else int(expected_count)
        self.spec = dict(json_safe(spec or {}))
        self.resume = bool(resume)
        self.overwrite = bool(overwrite)
        self.compression = compression
        self.compression_opts = compression_opts
        self.partial_path = self.path.with_name(self.path.name + ".partial")
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Any | None = None
        self._completed = False
        self._count = 0
        self._lock_owned = False

        if not overwrite and is_shard_complete(
            self.path, expected_count=self.expected_count, verify_checksum=True
        ):
            manifest = read_shard_manifest(self.path)
            if "spec" not in manifest or not _specs_match(manifest["spec"], self.spec):
                raise IncompleteShardError(
                    f"completed shard {self.path} belongs to a different job spec"
                )
            if not self._valid_finished_file(self.path, manifest_spec=manifest["spec"]):
                raise IncompleteShardError(
                    f"completed shard {self.path} contents do not match its requested job spec"
                )
            self._completed = True
            self._count = int(manifest["sample_count"])
            return

        self._acquire_lock()

        try:
            if self.path.exists():
                if overwrite:
                    # os.replace at finalize is recoverable until this point and no
                    # recursive or broad deletion is ever performed.
                    pass
                elif resume and self._valid_finished_file(self.path, adopt_stored_spec=True):
                    self._publish_sidecars()
                    self._completed = True
                    self._release_lock()
                    return
                else:
                    raise IncompleteShardError(
                        f"{self.path} exists without a valid completion record; "
                        "inspect it or pass overwrite=True"
                    )

            mode = "a" if resume and not overwrite and self.partial_path.exists() else "w"
            self._file = module.File(self.partial_path, mode)
            if mode == "w":
                self._file.attrs["schema_version"] = SCHEMA_VERSION
                self._file.attrs["spec_json"] = json.dumps(
                    self.spec, sort_keys=True, separators=(",", ":")
                )
                if self.expected_count is not None:
                    self._file.attrs["expected_count"] = self.expected_count
            else:
                self._validate_partial_header()
                self._repair_partial_rows()
            self._count = self._dataset_count()
        except BaseException:
            self._release_lock()
            raise

    def _acquire_lock(self) -> None:
        """Claim this shard exclusively before opening its partial HDF5 file."""

        owner = {
            "pid": os.getpid(),
            "hostname": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            try:
                existing = self.lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing = "<unreadable>"
            raise StorageError(
                f"shard is already locked by another generator: {self.lock_path}; "
                f"owner={existing}. Remove the lock only after confirming that job is no longer running."
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(owner, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            self.lock_path.unlink(missing_ok=True)
            raise
        self._lock_owned = True

    def _release_lock(self) -> None:
        if self._lock_owned:
            self.lock_path.unlink(missing_ok=True)
            self._lock_owned = False

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def count(self) -> int:
        return self._count

    def _validate_partial_header(self) -> None:
        assert self._file is not None
        if self._file.attrs.get("schema_version") != SCHEMA_VERSION:
            raise IncompleteShardError("partial shard schema version differs")
        try:
            existing_spec = json.loads(self._file.attrs.get("spec_json", "{}"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise IncompleteShardError("partial shard has an invalid job spec") from exc
        if not _specs_match(existing_spec, self.spec):
            raise IncompleteShardError("partial shard belongs to a different job spec")
        # Keep the original full audit record so HDF5 and completion sidecars
        # remain exactly consistent after resuming with fresh provenance.
        self.spec = dict(json_safe(existing_spec))
        existing_count = self._file.attrs.get("expected_count")
        if self.expected_count is not None and existing_count is not None:
            if int(existing_count) != self.expected_count:
                raise IncompleteShardError("partial shard expected_count differs")

    def _repair_partial_rows(self) -> None:
        assert self._file is not None
        present = [name for name in self.DATASET_NAMES if name in self._file]
        if not present:
            return
        if set(present) != set(self.DATASET_NAMES):
            raise IncompleteShardError("partial shard is missing required datasets")
        lengths = [len(self._file[name]) for name in self.DATASET_NAMES]
        complete_rows = min(lengths)
        for name, length in zip(self.DATASET_NAMES, lengths, strict=True):
            if length != complete_rows:
                self._file[name].resize(complete_rows, axis=0)
        self._file.flush()

    def _dataset_count(self) -> int:
        if self._file is None or "condition" not in self._file:
            return 0
        return int(len(self._file["condition"]))

    def _initialize(self, sample: Sample) -> None:
        module = _require_h5py()
        assert self._file is not None
        arrays = {
            "condition": sample.condition,
            "trajectory": sample.trajectory,
            "geometry": sample.geometry,
        }
        for name, array in arrays.items():
            chunk_shape = (1, *array.shape)
            self._file.create_dataset(
                name,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                chunks=chunk_shape,
                dtype=array.dtype,
                compression=self.compression,
                compression_opts=(self.compression_opts if self.compression is not None else None),
                shuffle=self.compression is not None,
            )
        self._file.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            chunks=(max(1, min(self.expected_count or 64, 256)),),
            dtype=module.string_dtype(encoding="utf-8"),
        )

    def append(self, sample: Sample) -> int:
        if self._completed:
            raise StorageError("cannot append to an already completed shard")
        if self._file is None:
            raise StorageError("writer is closed")
        if not isinstance(sample, Sample):
            raise TypeError("append expects a pdeobs.schema.Sample")
        if self.expected_count is not None and self._count >= self.expected_count:
            raise StorageError("append would exceed expected_count")
        if "condition" not in self._file:
            self._initialize(sample)
        for name, array in (
            ("condition", sample.condition),
            ("trajectory", sample.trajectory),
            ("geometry", sample.geometry),
        ):
            dataset = self._file[name]
            if tuple(dataset.shape[1:]) != tuple(array.shape):
                raise ValueError(f"{name} shape changed from {dataset.shape[1:]} to {array.shape}")
            dataset.resize(self._count + 1, axis=0)
            dataset[self._count] = array
        metadata = self._file["metadata"]
        metadata.resize(self._count + 1, axis=0)
        metadata[self._count] = json.dumps(
            json_safe(sample.metadata), sort_keys=True, separators=(",", ":")
        )
        self._count += 1
        self._file.flush()
        return self._count - 1

    def append_many(self, samples: Iterable[Sample]) -> int:
        for sample in samples:
            self.append(sample)
        return self._count

    def _valid_finished_file(
        self,
        path: Path,
        *,
        manifest_spec: Any | None = None,
        adopt_stored_spec: bool = False,
    ) -> bool:
        module = _require_h5py()
        try:
            with module.File(path, "r") as handle:
                required = set(self.DATASET_NAMES)
                if not required.issubset(handle):
                    return False
                lengths = {len(handle[name]) for name in required}
                if len(lengths) != 1:
                    return False
                count = lengths.pop()
                if self.expected_count is not None and count != self.expected_count:
                    return False
                if handle.attrs.get("schema_version") != SCHEMA_VERSION:
                    return False
                existing_spec = json.loads(handle.attrs.get("spec_json", "{}"))
                if not _specs_match(existing_spec, self.spec):
                    return False
                if manifest_spec is not None and not _full_specs_match(
                    existing_spec, manifest_spec
                ):
                    return False
                if adopt_stored_spec:
                    self.spec = dict(json_safe(existing_spec))
                self._count = int(count)
                return True
        except (OSError, TypeError, json.JSONDecodeError):
            return False

    def _read_metadata(self) -> list[dict[str, Any]]:
        module = _require_h5py()
        rows: list[dict[str, Any]] = []
        with module.File(self.path, "r") as handle:
            for encoded in handle["metadata"]:
                if isinstance(encoded, bytes):
                    encoded = encoded.decode("utf-8")
                rows.append(json.loads(str(encoded)))
        return rows

    def _publish_sidecars(self) -> dict[str, Any]:
        sidecars = shard_sidecars(self.path)
        rows = self._read_metadata()
        self._count = len(rows)

        atomic_write_json(
            sidecars["metadata_json"],
            {
                "schema_version": SCHEMA_VERSION,
                "shard": self.path.name,
                "samples": rows,
            },
        )
        flattened = [_flatten_metadata(row) for row in rows]
        fieldnames = sorted({key for row in flattened for key in row})
        csv_lines: list[str] = []
        if fieldnames:
            import io

            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flattened)
            csv_lines.append(buffer.getvalue())
        _atomic_text(sidecars["metadata_csv"], "".join(csv_lines))

        checksum = sha256_file(self.path)
        _atomic_text(sidecars["checksum"], f"{checksum}  {self.path.name}\n")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "shard": self.path.name,
            "sample_count": self._count,
            "bytes": self.path.stat().st_size,
            "sha256": checksum,
            "spec": self.spec,
            "metadata_csv": sidecars["metadata_csv"].name,
            "metadata_json": sidecars["metadata_json"].name,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(sidecars["manifest"], manifest)
        return manifest

    def finalize(self) -> dict[str, Any]:
        if self._completed:
            return read_shard_manifest(self.path)
        if self._file is None:
            raise StorageError("writer is closed")
        if self.expected_count is not None and self._count != self.expected_count:
            raise IncompleteShardError(
                f"cannot finalize {self._count} rows; expected {self.expected_count}"
            )
        if self._count == 0:
            raise IncompleteShardError("cannot finalize an empty shard")
        self._file.attrs["sample_count"] = self._count
        self._file.flush()
        self._file.close()
        self._file = None
        try:
            os.replace(self.partial_path, self.path)
            manifest = self._publish_sidecars()
            self._completed = True
        finally:
            self._release_lock()
        return manifest

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        self._release_lock()

    def __enter__(self) -> AtomicHDF5ShardWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None and not self._completed:
            try:
                self.finalize()
            except BaseException:
                self.close()
                raise
        else:
            self.close()
        return False


class LazyHDF5Dataset(Sequence[Sample]):
    """Index one or many shards without loading their arrays into memory."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        verify: bool = False,
    ) -> None:
        module = _require_h5py()
        del module
        if isinstance(paths, (str, Path)):
            candidates = [Path(paths)]
        else:
            candidates = [Path(path) for path in paths]
        expanded: list[Path] = []
        for candidate in candidates:
            if candidate.is_dir():
                expanded.extend(sorted(candidate.rglob("*.h5")))
                expanded.extend(sorted(candidate.rglob("*.hdf5")))
            else:
                expanded.append(candidate)
        self.paths = tuple(dict.fromkeys(path.resolve() for path in expanded))
        if not self.paths:
            raise ValueError("no HDF5 shards found")
        self.lengths: list[int] = []
        for path in self.paths:
            if verify and not is_shard_complete(path):
                raise IncompleteShardError(f"shard is not verified: {path}")
            with _require_h5py().File(path, "r") as handle:
                if not all(name in handle for name in AtomicHDF5ShardWriter.DATASET_NAMES):
                    raise StorageError(f"missing canonical datasets in {path}")
                self.lengths.append(int(len(handle["condition"])))
        self.offsets = np.cumsum([0, *self.lengths]).tolist()
        self._open_index: int | None = None
        self._handle: Any | None = None

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.offsets, index) - 1
        return shard_index, index - self.offsets[shard_index]

    def _open(self, shard_index: int) -> Any:
        if self._open_index != shard_index:
            self.close()
            self._handle = _require_h5py().File(self.paths[shard_index], "r")
            self._open_index = shard_index
        return self._handle

    def __getitem__(self, index: int | slice) -> Sample | list[Sample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        shard_index, local_index = self._locate(int(index))
        handle = self._open(shard_index)
        encoded = handle["metadata"][local_index]
        if isinstance(encoded, bytes):
            encoded = encoded.decode("utf-8")
        return Sample(
            condition=np.asarray(handle["condition"][local_index]),
            trajectory=np.asarray(handle["trajectory"][local_index]),
            geometry=np.asarray(handle["geometry"][local_index]),
            metadata=json.loads(str(encoded)),
        )

    def iter_metadata(self) -> Iterator[dict[str, Any]]:
        module = _require_h5py()
        for path in self.paths:
            with module.File(path, "r") as handle:
                for encoded in handle["metadata"]:
                    if isinstance(encoded, bytes):
                        encoded = encoded.decode("utf-8")
                    yield json.loads(str(encoded))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._open_index = None

    def __enter__(self) -> LazyHDF5Dataset:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_open_index"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - best effort at interpreter exit
        if hasattr(self, "_handle"):
            self.close()
