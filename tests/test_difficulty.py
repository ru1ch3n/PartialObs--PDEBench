import json

import pytest

from pdeobs.cli import main
from pdeobs.difficulty import AnalysisConfig, analyze_path, analyze_records
from pdeobs.reports import write_json_report


def _records():
    shared = {
        "method": "unet",
        "task": "rollout",
        "pde": "heat",
        "boundary": "periodic",
        "setting": "smooth_grf",
        "ood_view": "parameter",
        "parameter_count": 1_000,
    }
    values = [
        ("s0", "iid", "low", "random", 0.10, 100, 2.0, 0.2, 0.10),
        ("s1", "ood", "high", "random", 0.10, 100, 2.2, 0.4, 0.20),
        ("s2", "iid", "low", "grid", 0.20, 200, 3.0, 0.3, 0.15),
        ("s3", "ood", "high", "grid", 0.20, 200, 3.4, 0.8, 0.50),
    ]
    records = []
    for sample_id, split, regime, pattern, ratio, train_size, runtime, error, spectral in values:
        records.append(
            {
                **shared,
                "sample_id": sample_id,
                "split": split,
                "regime": regime,
                "mask_protocol": pattern,
                "observation_ratio": ratio,
                "train_size": train_size,
                "runtime_seconds": runtime,
                "metrics": {
                    "rel_l2": error,
                    "rel_l2_h1": error / 2,
                    "rel_l2_h4": error,
                    "spectral": {"high_band_error": spectral},
                },
            }
        )
    return records


def test_analysis_covers_difficulty_axes_and_is_deterministic(tmp_path):
    source = tmp_path / "metrics.json"
    write_json_report(_records(), source)
    output = tmp_path / "difficulty.json"

    report = analyze_path(
        source,
        output,
        config=AnalysisConfig(primary_metric="rel_l2", top_k=3),
    )
    first_bytes = output.read_bytes()
    analyze_path(
        source,
        output,
        config=AnalysisConfig(primary_metric="rel_l2", top_k=3),
    )

    assert output.read_bytes() == first_bytes
    assert report["schema_version"] == "pdeobs.problem-difficulty/v1"
    assert report["record_count"] == 4
    assert len(report["summaries"]["by_dimension"]["observation_ratio"]) == 2
    assert len(report["summaries"]["by_dimension"]["observation_pattern"]) == 2
    assert report["summaries"]["by_dimension"]["pde"][0]["pde"] == "heat"

    rollout = report["summaries"]["rollout_horizon"]
    assert [row["rollout_horizon"] for row in rollout] == [1, 4]
    assert rollout[1]["rel_l2.mean"] == pytest.approx(0.425)
    assert report["summaries"]["spectral"][0]["spectral_high_band_error.mean"] == pytest.approx(
        0.2375
    )

    degradation = [
        row for row in report["summaries"]["ood_degradation"] if row["metric"] == "rel_l2"
    ][0]
    assert degradation["iid_mean"] == pytest.approx(0.25)
    assert degradation["ood_mean"] == pytest.approx(0.6)
    assert degradation["absolute_degradation"] == pytest.approx(0.35)

    scaling = report["summaries"]["scaling"]
    training_levels = [
        row for row in scaling["levels"] if row["scale_variable"] == "training_samples"
    ]
    assert [row["scale_value"] for row in training_levels] == [100.0, 200.0]
    assert any(
        row["scale_variable"] == "training_samples" and row["metric"] == "rel_l2"
        for row in scaling["trends"]
    )

    assert report["failure_rankings"]["records"][0]["sample_id"] == "s3"
    assert report["failure_rankings"]["records"][0]["value"] == pytest.approx(0.8)
    for suffix in (
        ".dimensions.csv",
        ".rollout.csv",
        ".spectral.csv",
        ".ood.csv",
        ".scaling.csv",
        ".scaling-trends.csv",
        ".failures.csv",
    ):
        assert (tmp_path / f"difficulty{suffix}").is_file()


def test_cli_and_long_form_records(tmp_path, capsys):
    source = tmp_path / "metrics.csv"
    source.write_text(
        "method,task,pde,metric,value,observation_fraction\n"
        "nearest,recovery,poisson,mse,0.4,10%\n"
        "nearest,recovery,poisson,mse,0.2,20%\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"

    assert (
        main(
            [
                "analyze",
                "--input",
                str(source),
                "--output",
                str(output_dir),
                "--primary-metric",
                "mse",
                "--top-k",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads((output_dir / "difficulty.json").read_text(encoding="utf-8"))
    ratios = payload["summaries"]["by_dimension"]["observation_ratio"]
    assert [row["observation_ratio"] for row in ratios] == [0.1, 0.2]
    assert payload["failure_rankings"]["records"][0]["value"] == pytest.approx(0.4)
    assert "Analyzed 2 records" in capsys.readouterr().out


def test_analysis_rejects_records_without_metrics():
    with pytest.raises(ValueError, match="No numeric metric fields"):
        analyze_records([{"pde": "heat", "regime": "low"}])
