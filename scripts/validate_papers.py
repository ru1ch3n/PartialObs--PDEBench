"""Validate JSON curation files under data/curations/.

This is a lightweight schema checker to catch common mistakes early
(wrong types, missing required fields, etc.).

Usage:
    python scripts/validate_papers.py

Exit code:
    0: no errors
    1: errors found
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CURATIONS_DIR = REPO_ROOT / "data" / "curations"


def _as_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def _as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def validate_one(path: Path) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return ([f"JSON parse error: {e}"], [])

    if obj is None:
        return (["File is empty (JSON is null)."], [])
    if not isinstance(obj, dict):
        return ([f"Top-level JSON must be an object/dict, got {type(obj).__name__}."], [])

    slug_file = path.stem
    slug = str(obj.get("slug") or slug_file)
    if slug != slug_file:
        warnings.append(f"slug '{slug}' does not match filename '{slug_file}'.")

    # Title fields
    title = obj.get("full_title") or obj.get("title") or obj.get("short_title")
    if not title or not str(title).strip():
        errors.append("Missing title: set full_title (preferred) or title.")

    # Year
    year = obj.get("year")
    if year is None:
        warnings.append("Missing year.")
    else:
        try:
            int(year)
        except Exception:
            errors.append(f"year must be an integer (or int-like string), got {year!r}.")

    # Status
    status = str(obj.get("status") or "index")
    if status not in {"index", "curated"}:
        warnings.append("status should be 'index' or 'curated'.")

    # Lists
    for key in ["pdes", "tasks", "setting", "baselines", "contrib", "theory", "core_math", "data_setting", "model_setting", "training_setting", "interesting", "benefits"]:
        if key in obj and obj[key] is not None and not isinstance(obj[key], list):
            errors.append(f"{key} must be a list, got {type(obj[key]).__name__}.")

    # auto block
    auto = _as_dict(obj.get("auto"))
    for key in ["pdes", "tasks"]:
        if key in auto and auto[key] is not None and not isinstance(auto[key], list):
            errors.append(f"auto.{key} must be a list, got {type(auto[key]).__name__}.")

    # results_tables format
    rt = obj.get("results_tables")
    if rt is not None:
        if not isinstance(rt, list):
            errors.append(f"results_tables must be a list, got {type(rt).__name__}.")
        else:
            for i, t in enumerate(rt):
                if not isinstance(t, dict):
                    errors.append(f"results_tables[{i}] must be a dict, got {type(t).__name__}.")
                    continue
                header = t.get("header")
                rows = t.get("rows")
                if header is not None and not isinstance(header, list):
                    errors.append(f"results_tables[{i}].header must be a list.")
                if rows is not None and not isinstance(rows, list):
                    errors.append(f"results_tables[{i}].rows must be a list (of row lists).")
                for j, r in enumerate(_as_list(rows)):
                    if not isinstance(r, list):
                        errors.append(f"results_tables[{i}].rows[{j}] must be a list.")

    return (errors, warnings)


def main() -> int:
    if not CURATIONS_DIR.exists():
        print(f"No curations directory: {CURATIONS_DIR}")
        return 1

    files = sorted([p for p in CURATIONS_DIR.glob("*.json") if not p.name.startswith("_")])
    if not files:
        print(f"No JSON curation files found in {CURATIONS_DIR}")
        return 1

    n_err = 0
    n_warn = 0

    for path in files:
        errors, warnings = validate_one(path)
        if errors or warnings:
            rel = path.relative_to(REPO_ROOT)
            if errors:
                n_err += 1
                print(f"[ERROR] {rel}")
                for e in errors:
                    print(f"  - {e}")
            if warnings:
                n_warn += 1
                print(f"[WARN ] {rel}")
                for w in warnings:
                    print(f"  - {w}")

    print(f"\nChecked {len(files)} JSON files: {n_err} with errors, {n_warn} with warnings.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
