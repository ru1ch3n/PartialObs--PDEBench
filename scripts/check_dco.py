# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Verify Developer Certificate of Origin sign-offs for a commit range."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass

_SIGNOFF_RE = re.compile(r"^Signed-off-by:\s*(?P<name>.+?)\s*<(?P<email>[^>]+)>\s*$", re.I)


@dataclass(frozen=True)
class CommitRecord:
    """The identity and message needed for a DCO check."""

    sha: str
    author_name: str
    author_email: str
    message: str


def signoff_emails(message: str) -> set[str]:
    """Return normalized emails from valid Signed-off-by trailers."""

    return {
        match.group("email").strip().casefold()
        for line in message.splitlines()
        if (match := _SIGNOFF_RE.match(line.strip()))
    }


def commit_is_signed(record: CommitRecord) -> bool:
    """Accept a commit only when the author's email has a matching sign-off."""

    return record.author_email.strip().casefold() in signoff_emails(record.message)


def commits_in_range(commit_range: str) -> list[CommitRecord]:
    """Read commit records from Git without relying on locale-sensitive output."""

    field = "%x1f"
    record = "%x1e"
    completed = subprocess.run(
        [
            "git",
            "log",
            "--format=%H" + field + "%an" + field + "%ae" + field + "%B" + record,
            commit_range,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commits: list[CommitRecord] = []
    for raw_record in completed.stdout.split("\x1e"):
        raw_record = raw_record.strip()
        if not raw_record:
            continue
        parts = raw_record.split("\x1f", 3)
        if len(parts) != 4:
            raise RuntimeError("Could not parse git log output for DCO verification")
        commits.append(CommitRecord(*parts))
    return commits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit_range", help="Git revision range, for example base..head")
    args = parser.parse_args()

    commits = commits_in_range(args.commit_range)
    failures = [record for record in commits if not commit_is_signed(record)]
    if failures:
        for item in failures:
            print(
                f"DCO failure: {item.sha[:12]} by {item.author_name} "
                f"<{item.author_email}> lacks a matching Signed-off-by trailer"
            )
        return 1
    print(f"DCO check passed for {len(commits)} commit(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
