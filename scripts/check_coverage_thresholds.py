# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Enforce statement and branch coverage independently from coverage.py JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def coverage_percentages(report: dict[str, Any]) -> tuple[float, float]:
    """Calculate statement and branch percentages from a coverage.py report."""

    totals = report["totals"]
    statements = int(totals["num_statements"])
    missing_statements = int(totals["missing_lines"])
    branches = int(totals.get("num_branches", 0))
    missing_branches = int(totals.get("missing_branches", 0))
    statement_percent = (
        100.0 if statements == 0 else (statements - missing_statements) * 100 / statements
    )
    branch_percent = 100.0 if branches == 0 else (branches - missing_branches) * 100 / branches
    return statement_percent, branch_percent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-statements", type=float, required=True)
    parser.add_argument("--min-branches", type=float, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    statement_percent, branch_percent = coverage_percentages(report)
    print(f"statement coverage: {statement_percent:.2f}% (minimum {args.min_statements:.2f}%)")
    print(f"branch coverage: {branch_percent:.2f}% (minimum {args.min_branches:.2f}%)")
    if statement_percent < args.min_statements or branch_percent < args.min_branches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
