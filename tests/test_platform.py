"""Published evidence, provenance and static navigation contracts."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLATFORM = json.loads((ROOT / "data/platform.json").read_text(encoding="utf-8"))
SNAPSHOT = DOCS / "assets/snapshots/paper-20260925-v1.json"


def test_snapshot_preserves_complete_reference_and_provenance() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    manifest = json.loads((SNAPSHOT.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    assert manifest["sha256"] == "150e6e634caa7b2bc46437833fac6d9160ea6cf03f179f0d5aa5d122a7778448"
    assert snapshot["source_commit"] == "ab62fa7504627c124c187e1822b27b95e79c9a76"
    assert snapshot["source_commit"] in snapshot["source_url"]
    assert len(snapshot["source_sha256"]) == 64
    assert snapshot["compute_cost"] is None
    assert "did not rerun inference or training" in snapshot["verification_scope"]
    records = snapshot["records"]
    assert len(records) == 441
    counts = Counter()
    views = {v["id"] for v in PLATFORM["views"]}
    for identity, record in records.items():
        assert identity == f"{record['pde']}/{record['method']}/{record['train_view']}"
        assert len(record["checkpoint_released_sha256"]) == 64
        assert set(record["blocks"]) == views
        assert record["actual_epochs"] > 0
        for block in record["blocks"].values():
            counts[record["method"]] += 1
            assert block["joint"]["n"] == 200
            assert block["joint"]["ddof"] == 1
            for metric in ("mean", "std"):
                assert math.isfinite(block["joint"][metric])
                assert block["joint"][metric] >= 0
            assert len(block["contract_sha256"]) == 64
            assert all(len(value) == 64 for value in block["file_hashes"].values())
    assert len(counts) == 7
    assert set(counts.values()) == {567}
    assert sum(counts.values()) == 3969


def test_reference_matrix_links_to_exact_snapshot_values() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    soup = BeautifulSoup((DOCS / "index.html").read_text(encoding="utf-8"), "html.parser")
    cells = soup.select(".matrix td a")
    assert len(cells) == 81
    for cell in cells:
        params = parse_qs(urlsplit(cell["href"]).query)
        identity = "/".join(params[key][0] for key in ("pde", "method", "train"))
        score = snapshot["records"][identity]["blocks"][params["test"][0]]["joint"]["mean"]
        assert cell.get_text() == f"{score:.2f}"


def test_platform_internal_links_and_fragments_resolve() -> None:
    pages = [DOCS / "index.html"] + [
        DOCS / name / "index.html"
        for name in (
            "benchmark",
            "results",
            "methods",
            "studies",
            "releases",
            "contribute",
            "guide",
            "research",
        )
    ]
    for page in pages:
        soup = BeautifulSoup(page.read_text(encoding="utf-8"), "html.parser")
        assert len(soup.select("h1")) == 1
        ids = [tag["id"] for tag in soup.select("[id]")]
        assert len(ids) == len(set(ids)), page
        for element in soup.select("a[href], script[src], link[href]"):
            value = element.get("href", element.get("src"))
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            target = (page.parent / url.path).resolve() if url.path else page
            if target.is_dir():
                target /= "index.html"
            assert target.is_file(), (page, value)
            assert target.is_relative_to(DOCS.resolve())
            if url.fragment and target.suffix == ".html":
                other = BeautifulSoup(target.read_text(encoding="utf-8"), "html.parser")
                assert other.find(id=url.fragment), (page, value)


def test_contribution_routes_have_real_required_issue_forms() -> None:
    page = BeautifulSoup(
        (DOCS / "contribute/index.html").read_text(encoding="utf-8"), "html.parser"
    )
    forms = set()
    for anchor in page.select(".action-card a"):
        url = urlsplit(anchor["href"])
        assert url.path == "/ru1ch3n/PartialObs--PDEBench/issues/new"
        forms.add(parse_qs(url.query)["template"][0])
    assert forms == {"submit_method.yml", "reproduce_result.yml", "join_study.yml"}
    for filename in forms:
        form = yaml.safe_load(
            (ROOT / ".github/ISSUE_TEMPLATE" / filename).read_text(encoding="utf-8")
        )
        fields = [field for field in form["body"] if field["type"] != "markdown"]
        assert len({field["id"] for field in fields}) == len(fields)
        assert sum(field.get("validations", {}).get("required", False) for field in fields) >= 3


def test_no_unreleased_results_or_official_adapter_claims() -> None:
    assert all(study["status"].startswith("Proposed") for study in PLATFORM["studies"])
    methods = {row["id"]: row for row in PLATFORM["methods"]}
    assert methods["cno"]["kind"] == "Inspired implementation"
    assert all(row["kind"] != "Official implementation" for row in methods.values())
    result_html = (DOCS / "results/index.html").read_text(encoding="utf-8")
    assert "No community evaluation has been published yet" in result_html
    assert "No study results or completed experiments are claimed" in result_html
    assert "not an independent rerun" in result_html
