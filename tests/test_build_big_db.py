from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_big_db.py"
SPEC = importlib.util.spec_from_file_location("pdeobs_build_big_db", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD_BIG_DB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD_BIG_DB
SPEC.loader.exec_module(BUILD_BIG_DB)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/openai/example",
        "https://www.github.com/openai/example",
        "https://GITHUB.COM/openai/example",
        "https://github.com./openai/example",
    ],
)
def test_is_github_url_accepts_only_canonical_https_hosts(url: str) -> None:
    assert BUILD_BIG_DB.is_github_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://evilgithub.com/openai/example",
        "https://github.com.evil.example/openai/example",
        "https://example.com/?next=github.com/openai/example",
        "http://github.com/openai/example",
        "javascript://github.com/%0Aalert(1)",
        "not a url containing github.com",
    ],
)
def test_is_github_url_rejects_lookalike_and_non_https_urls(url: str) -> None:
    assert not BUILD_BIG_DB.is_github_url(url)
