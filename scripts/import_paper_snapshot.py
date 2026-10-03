"""Import an explicitly pinned public result index into the website, read-only.

Run only when deliberately adding a NEW snapshot. Never refresh an existing one.
The website build is offline and does not run this importer.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Snapshot already exists; use a new version, never overwrite.")
    commit = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", args.commit], text=True
    ).strip()
    base = "results/prediction_verification_20260924"
    raw = subprocess.check_output(
        ["git", "-C", str(args.source), "show", f"{commit}:{base}/index.json"]
    )
    source = json.loads(raw)
    result = {
        "id": "paper-20260925-v1",
        "schema_version": "pdeobs-website-snapshot/v1",
        "source_schema": source["schema_version"],
        "source_commit": commit,
        "source_url": f"https://github.com/ru1ch3n/PDE-OBS/blob/{commit}/{base}/index.json",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "audit_url": f"https://github.com/ru1ch3n/PDE-OBS/blob/{commit}/{base}/README.md",
        "dataset_bindings_url": f"https://github.com/ru1ch3n/PDE-OBS/blob/{commit}/{base}/dataset_bindings.json",
        "manifest_url": f"https://github.com/ru1ch3n/PDE-OBS/blob/{commit}/{base}/manifest.json",
        "evaluator_version": "pdeobs-strict-v1",
        "verification": "Artifacts-checked (source audit)",
        "verification_scope": "Source audit checked configurations, identities, masks, physical times, file hashes and prediction scoring. It did not rerun inference or training. Website import is a transcription, not independent verification.",
        "metric": "Mean per-record joint relative L2; sample SD (ddof=1), 200 physical records per block. Forecasting uses joint three-frame error. SD is not seed uncertainty or a confidence interval.",
        "compute_cost": None,
        "compute_cost_note": "Comparable training time, inference latency and memory are not available in this snapshot. Audit seconds are not inference cost.",
        "summary": source["summary"],
        "records": {},
    }
    for identity, record in source["records"].items():
        copied = {
            k: record[k]
            for k in (
                "identity",
                "pde",
                "method",
                "task",
                "train_view",
                "actual_epochs",
                "training_cohort",
                "checkpoint_released_sha256",
                "provenance_sha256",
                "training_config",
                "model_file_hashes",
                "cohort_evidence",
            )
        }
        copied["blocks"] = {
            view: {k: block[k] for k in ("joint", "horizons", "contract_sha256", "file_hashes")}
            for view, block in record["blocks"].items()
        }
        result["records"][identity] = copied
    assert len(result["records"]) == 441
    assert sum(len(r["blocks"]) for r in result["records"].values()) == 3969
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Imported {len(result['records'])} models from {commit}; source unchanged.")


if __name__ == "__main__":
    main()
