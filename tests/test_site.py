from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_generated_pages_include_run_navigation() -> None:
    pages = sorted(DOCS.rglob("index.html"))
    assert pages

    missing = []
    for page in pages:
        html = page.read_text(encoding="utf-8")
        if '<nav class="nav">' not in html or ">Run</a>" not in html:
            missing.append(str(page.relative_to(ROOT)))

    assert not missing, f"Run navigation missing from: {missing}"


def test_server_page_contains_both_supported_workflows() -> None:
    html = (DOCS / "server" / "index.html").read_text(encoding="utf-8")

    assert "Linux server: verified smoke run" in html
    assert "SeaWulf: dependency-chained smoke example" in html
    assert "configs/dataset/smoke.yaml" in html
    assert "hpc/seawulf/generate_array.sbatch" in html
    assert 'aria-current="page">Run</a>' in html


def test_public_benchmark_page_uses_the_benchmark_only_scope() -> None:
    html = (DOCS / "benchmark" / "index.html").read_text(encoding="utf-8")

    assert "PDE-OBS benchmark paper" in html
    assert "only manuscript in scope" in html
    assert "7 task protocols" in html
    assert "15 analyses" in html
    assert "pdeobs protocol --check" in html
    assert "pdeobs generate --tier signal" in html
    assert "docs/BENCHMARK_PAPER.md" in html


def test_generated_pages_cache_bust_the_shared_stylesheet() -> None:
    pages = sorted(DOCS.rglob("index.html"))
    missing = [
        str(page.relative_to(ROOT))
        for page in pages
        if "assets/style.css?v=" not in page.read_text(encoding="utf-8")
    ]

    assert not missing, f"Stylesheet cache buster missing from: {missing}"


def test_homepage_inline_mermaid_script_is_not_truncated_by_a_line_comment() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")

    assert "/* Enable clickable Mermaid nodes on GitHub Pages. */" in html
    assert "// Enable clickable nodes" not in html
