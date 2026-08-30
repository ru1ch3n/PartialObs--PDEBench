# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Check or add project copyright and SPDX headers to source files."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

COPYRIGHT = "Copyright 2026 PDE-OBS contributors"
SPDX = "SPDX-License-Identifier: MIT"
ROOT = Path(__file__).resolve().parents[1]


def source_files(root: Path = ROOT) -> Iterable[Path]:
    """Yield maintained source/configuration files covered by the policy."""

    roots = (
        root / ".github",
        root / ".clusterfuzzlite",
        root / "configs",
        root / "hpc",
        root / "scripts",
        root / "src",
        root / "tests",
        root / "docs",
    )
    allowed_suffixes = {".py", ".sh", ".sbatch", ".yml", ".yaml", ".html", ".css", ".js", ".svg"}
    for directory in roots:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (
                path.suffix.lower() in allowed_suffixes or path.name == "Dockerfile"
            ):
                yield path
    for name in ("pyproject.toml", "environment.yml"):
        path = root / name
        if path.is_file():
            yield path


def comment_prefix(path: Path) -> str:
    if path.suffix.lower() in {".html", ".svg"}:
        return "xml"
    if path.suffix.lower() in {".css", ".js"}:
        return "slash"
    return "hash"


def header_lines(path: Path) -> list[str]:
    kind = comment_prefix(path)
    if kind == "xml":
        return [f"<!-- {COPYRIGHT} -->\n", f"<!-- {SPDX} -->\n"]
    if kind == "slash":
        return [f"/* {COPYRIGHT} */\n", f"/* {SPDX} */\n"]
    return [f"# {COPYRIGHT}\n", f"# {SPDX}\n"]


def insert_header(path: Path, text: str) -> str:
    """Insert after an interpreter/XML declaration, otherwise at the beginning."""

    lines = text.splitlines(keepends=True)
    index = 0
    if lines and (
        lines[0].startswith("#!")
        or lines[0].lstrip().startswith("<?xml")
        or lines[0].lstrip().lower().startswith("<!doctype html")
    ):
        index = 1
    return "".join(lines[:index] + header_lines(path) + lines[index:])


def has_header(path: Path, text: str) -> bool:
    """Require the two exact comment lines near the start, not string literals."""

    leading_lines = set(text.splitlines()[:8])
    expected_lines = {line.rstrip("\n") for line in header_lines(path)}
    return expected_lines <= leading_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    missing: list[Path] = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        if has_header(path, text):
            continue
        missing.append(path)
        if args.fix:
            path.write_text(insert_header(path, text), encoding="utf-8", newline="")
    if missing and not args.fix:
        for path in missing:
            print(path.relative_to(ROOT).as_posix())
        print(f"SPDX headers missing from {len(missing)} source file(s)")
        return 1
    action = "updated" if args.fix else "verified"
    print(f"SPDX headers {action} for {len(list(source_files()))} source file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
