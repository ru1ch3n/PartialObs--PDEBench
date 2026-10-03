import json
from pathlib import Path

from pdeobs.methods import available_methods
from pdeobs.protocol import benchmark_contract

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_generated_pages_include_run_and_builder_navigation() -> None:
    pages = sorted(DOCS.rglob("index.html"))
    assert pages

    missing = []
    for page in pages:
        html = page.read_text(encoding="utf-8")
        if '<nav class="nav">' not in html or ">Builder</a>" not in html or ">Run</a>" not in html:
            missing.append(str(page.relative_to(ROOT)))

    assert not missing, f"Builder/Run navigation missing from: {missing}"


def test_server_page_contains_both_supported_workflows() -> None:
    html = (DOCS / "server" / "index.html").read_text(encoding="utf-8")

    assert "Linux server: verified smoke run" in html
    assert "Slurm HPC: dependency-chained smoke example" in html
    assert "configs/dataset/smoke.yaml" in html
    assert "git checkout YOUR_RELEASE_TAG_OR_COMMIT" in html
    assert "hpc/slurm/generate_array.sbatch" in html
    assert "YOUR_CPU_PARTITION" in html
    assert "YOUR_GPU_PARTITION" in html
    assert 'href="../builder/">Benchmark Builder</a>' in html
    assert "OBSERVATION_TRAINING_PROTOCOL.md" in html
    assert "random 3% checkpoint belongs to a separate transfer/OOD table" in html
    assert 'aria-current="page">Run</a>' in html


def test_builder_page_is_bilingual_accessible_and_quality_explicit() -> None:
    html = (DOCS / "builder" / "index.html").read_text(encoding="utf-8")

    assert "Benchmark Builder / 基准构建器" in html
    assert "全 PDE 数据质量管理" in html
    assert 'id="benchmark-builder"' in html
    assert 'role="tablist"' in html
    assert 'aria-live="polite"' in html
    assert "benchmark-builder.js?v=" in html
    assert "benchmark-builder.js?v=2026-08-31-slurm-v1" in html
    assert "Training per observation type" in html
    assert "Primary matched-mask comparison" in html
    assert "random 3%" in html
    assert "mask-transfer/OOD" in html
    assert "OBSERVATION_TRAINING_PROTOCOL.md" in html
    assert 'id="campaign-preset"' in html
    assert 'id="campaign-method-body"' in html
    assert "Campaign plan (not executable)" in html
    assert "all seven losses together" in html
    assert "*.quality-failures.jsonl" in html
    assert "publication_ready" in html
    assert 'aria-current="page">Builder</a>' in html


def test_builder_options_are_generated_from_the_frozen_contract() -> None:
    payload = json.loads(
        (DOCS / "assets" / "benchmark-builder-options.json").read_text(encoding="utf-8")
    )
    contract = benchmark_contract()

    assert payload["generated_from"]["contract"] == contract["schema_version"]
    assert [row["value"] for row in payload["pdes"]] == contract["dataset"]["pde_families"]
    assert [row["value"] for row in payload["boundaries"]] == contract["dataset"]["boundaries"]
    assert [row["value"] for row in payload["settings"]] == contract["dataset"]["settings"]
    assert [row["value"] for row in payload["masks"]] == contract["masks"]
    assert {row["value"] for row in payload["quality_profiles"]} == {
        "report",
        "strict",
        "publication",
    }
    assert len(payload["pdes"]) == 7
    assert all(row["loss"] and row["note"] for row in payload["pdes"])
    assert any("summary.quality.json" in item for item in payload["quality_outputs"])
    assert any("*.quality-failures.jsonl" in item for item in payload["quality_outputs"])
    assert {row["value"] for row in payload["models"]} == set(available_methods())
    assert all(
        isinstance(row["capabilities_known"], bool)
        and isinstance(row["trainable"], bool)
        and isinstance(row["supports_multichannel"], bool)
        for row in payload["models"]
    )
    observation_training = contract["observation_training"]
    protocol_methods = payload["protocol_methods"]
    assert [row["method_id"] for row in protocol_methods] == [
        row["method_id"] for row in observation_training["methods"]
    ]
    assert len(protocol_methods) == 10
    assert all(
        row["builder_available"]
        == bool(row["registry_name"] and row["registry_name"] in available_methods())
        for row in protocol_methods
    )
    assert all(
        row["command_generation"] == "blocked"
        for row in protocol_methods
        if not row["builder_available"]
    )
    planner = payload["campaign_planner"]
    assert planner["primary"]["training_mask_equals_evaluation_mask"] is True
    assert planner["secondary"]["training_mask"] == "random_3pct"
    assert planner["secondary"]["separate_result_table"] is True
    presets = {row["value"]: row for row in planner["presets"]}
    medium = presets["medium_recommended"]
    assert (
        medium["result_cells"],
        medium["preparation_jobs_min"],
        medium["preparation_jobs_max"],
        medium["raw_evaluation_runs"],
    ) == (630, 511, 525, 756)
    assert medium["execution_status"] == "planning_only_blocked"
    assert (
        medium["method_observations"]["pinn_or_pino"]
        == contract["observation_training"]["primary"]["observation_protocols"]
    )
    assert (
        next(row for row in protocol_methods if row["value"] == "pinn_or_pino")["default_seeds"]
        == 3
    )
    assert {
        "gappy_pod",
        "deeponet",
        "pinn_or_pino",
        "transolver_or_gnot",
        "diffusionpde",
        "fundps",
    }.issubset(medium["blocked_methods"])
    full_anchor = presets["full_anchor_hybrid"]
    assert (
        full_anchor["result_cells"],
        full_anchor["preparation_jobs_min"],
        full_anchor["preparation_jobs_max"],
    ) == (231, 70, 84)
    assert (
        full_anchor["method_observations"]["rbf"]
        == contract["observation_training"]["primary"]["observation_protocols"]
    )
    assert full_anchor["method_observations"]["unet"] == [
        "random_1pct",
        "random_3pct",
        "block_missing",
    ]
    assert planner["dataset_accounting"]["medium"]["records_per_pde"] == 20_000
    assert planner["dataset_accounting"]["full"]["records_per_pde"] == 80_000
    assert planner["budget"]["status"] == "unmeasured_planning_scenario"
    assert "not a capacity promise for any Slurm system" in planner["budget"]["warning"]
    assert payload["environments"][0]["label"] == "Linux/macOS/local Bash"
    assert {row["value"] for row in payload["environments"]} == {"local", "server", "slurm"}


def test_builder_script_generates_safe_reproducible_workflows() -> None:
    script = (DOCS / "assets" / "benchmark-builder.js").read_text(encoding="utf-8")

    assert "innerHTML" not in script
    assert ".textContent" in script
    assert "URLSearchParams" in script
    assert "navigator.clipboard" in script
    assert "pdeobs plan" in script
    assert "pdeobs quality" in script
    assert "--require-all-pdes" in script
    assert "--require-validated-solvers" in script
    assert "submit_generation.sh" in script
    assert script.count("set -Eeuo pipefail") >= 3
    assert 'CONFIG_DIR="$PDEOBS_DATA/configs"' in script
    assert "PDEOBS_WINDOW_START" in script
    assert "PDEOBS_MAX_QUEUED_TASKS" in script
    assert "afterok:$generation_job" in script
    assert "summary.quality.json/.csv" in script
    assert "publication_ready remains false" in script
    assert "git checkout YOUR_RELEASE_TAG_OR_COMMIT" in script
    assert "factor-sweep training" in script
    assert "capabilities_known" in script
    assert "model.trainable" in script
    assert "PLANNING MANIFEST ONLY" in script
    assert '"method_plan:"' in script
    assert "preset.method_observations[row.value]" in script
    assert "seeds: ${row.default_seeds}" in script
    assert "fit_scope: ${yamlString(row.fit_scope)}" in script
    assert "No training, evaluation, or scheduler command is emitted" in script
    assert "slurm_site_transferable: false" in script
    assert "command_generation" in script
    assert 'state.model === "autoregressive"' in script
    for split in (
        "boundary_ood",
        "setting_ood",
        "parameter_ood",
        "combination_ood",
        "mask_ood",
        "time_horizon_ood",
    ):
        assert split in script
    assert "explicit experiment YAML" in script
    assert 'CONFIG="pdeobs-builder.yaml"' not in script
    assert "for ((start=0" not in script
    assert "previous_job" not in script
    assert "PowerShell alternative" not in script
    assert "dependency-chained windows" not in script
    assert "YOUR_CPU_PARTITION" in script
    assert "YOUR_GPU_PARTITION" in script
    assert "hpc/slurm/submit_generation.sh" in script
    assert "\\n+  " not in script


def test_site_generator_has_no_wall_clock_output_drift() -> None:
    source = (ROOT / "scripts" / "generate_research_site.py").read_text(encoding="utf-8")

    assert "datetime.now" not in source
    assert "Last generated:" not in source
    assert "Generated deterministically from repository sources." in source


def test_public_benchmark_page_uses_the_benchmark_only_scope() -> None:
    html = (DOCS / "benchmark" / "index.html").read_text(encoding="utf-8")

    assert "PDE-OBS benchmark paper" in html
    assert "only manuscript in scope" in html
    assert "7 task protocols" in html
    assert "15 analyses" in html
    assert "pdeobs protocol --check" in html
    assert "pdeobs generate --tier signal" in html
    assert "docs/BENCHMARK_PAPER.md" in html
    assert "Matched-mask training protocol" in html
    assert "630" in html
    assert "511-525" in html
    assert "OBSERVATION_TRAINING_PROTOCOL.md" in html


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
