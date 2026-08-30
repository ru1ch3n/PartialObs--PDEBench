# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from scripts.check_coverage_thresholds import coverage_percentages
from scripts.check_dco import CommitRecord, commit_is_signed, signoff_emails
from scripts.check_spdx_headers import COPYRIGHT, SPDX, insert_header


def test_dco_requires_a_signoff_matching_the_author_email() -> None:
    record = CommitRecord(
        sha="abc123",
        author_name="Example Contributor",
        author_email="contributor@example.org",
        message="Improve validation\n\nSigned-off-by: Example Contributor <contributor@example.org>",
    )
    assert commit_is_signed(record)
    assert signoff_emails(record.message) == {"contributor@example.org"}
    assert not commit_is_signed(
        CommitRecord(record.sha, record.author_name, record.author_email, "No trailer")
    )


def test_coverage_threshold_input_keeps_statements_and_branches_separate() -> None:
    report = {
        "totals": {
            "num_statements": 100,
            "missing_lines": 10,
            "num_branches": 50,
            "missing_branches": 10,
        }
    }
    assert coverage_percentages(report) == (90.0, 80.0)


def test_spdx_header_preserves_shebang() -> None:
    path = Path("worker.py")
    rendered = insert_header(path, "#!/usr/bin/env python\nprint('ok')\n")
    lines = rendered.splitlines()
    assert lines[0] == "#!/usr/bin/env python"
    assert COPYRIGHT in lines[1]
    assert SPDX in lines[2]


def test_spdx_header_preserves_html_doctype() -> None:
    rendered = insert_header(Path("index.html"), "<!doctype html>\n<html></html>\n")
    lines = rendered.splitlines()
    assert lines[0] == "<!doctype html>"
    assert COPYRIGHT in lines[1]
    assert SPDX in lines[2]
