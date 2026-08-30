# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Command-line interface for generation, training, evaluation, and SeaWulf."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import __version__


def _overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted YAML key; repeat as needed",
    )


def _experiment_selector(parser: argparse.ArgumentParser, *, include_model: bool = True) -> None:
    """Add the wheel-safe one-line preset selectors while retaining --config."""

    parser.add_argument("--task", help="paper task name, for example sparse_recovery")
    if include_model:
        parser.add_argument("--model", help="registered method name, for example fno")
    parser.add_argument("--data", type=Path, help="generated tier directory")
    parser.add_argument("--split", default=None, help="iid, an OOD view, or an exact split")
    parser.add_argument("--mask", default=None, help="mask protocol, for example random_3pct")
    parser.add_argument("--pde", help="optional reference-case PDE filter")
    parser.add_argument("--boundary", help="optional reference-case boundary filter")
    parser.add_argument("--setting", help="optional reference-case setting filter")
    parser.add_argument("--param-regime", dest="param_regime", help="optional regime filter")


def _quality_profile_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quality-profile",
        choices=("report", "strict", "publication"),
        help=(
            "dataset-quality policy: report measurements, reject failed calibrated "
            "checks, or require publication-grade solver/threshold evidence"
        ),
    )
    parser.add_argument(
        "--max-pde-loss",
        type=float,
        help="maximum normalized PDE loss (must be calibrated for the selected protocol)",
    )


def _quality_gate_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quality-strict",
        "--strict",
        action="store_true",
        help="fail on stored quality failures",
    )
    parser.add_argument(
        "--max-pde-loss",
        type=float,
        help="fail if any PDE family's maximum normalized loss exceeds this value",
    )
    parser.add_argument(
        "--require-all-pdes",
        action="store_true",
        help="require quality records for all seven built-in PDE families",
    )
    parser.add_argument(
        "--require-validated-solvers",
        action="store_true",
        help="require every present PDE to use a solver marked independently validated",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdeobs",
        description="PDE-OBS partial-observation PDE benchmark",
    )
    parser.add_argument("--version", action="version", version=f"pdeobs {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="verify runtime, storage, Slurm, and GPU setup")
    doctor.add_argument("--cluster", choices=("local", "seawulf"), default="local")
    doctor.add_argument("--gpu", action="store_true", help="require a usable PyTorch GPU")
    doctor.add_argument("--offline", action="store_true", help="assert an offline job setup")
    doctor.set_defaults(handler=_cmd_doctor)

    listing = commands.add_parser("list", help="list registered extensible components")
    listing.add_argument(
        "--kind",
        choices=("all", "pdes", "settings", "masks", "methods", "metrics"),
        default="all",
    )
    listing.add_argument("--plugins", action="store_true", help="discover installed entry points")
    listing.add_argument("--json", action="store_true", dest="as_json")
    listing.set_defaults(handler=_cmd_list)

    protocol = commands.add_parser(
        "protocol", help="print or validate the frozen benchmark-paper contract"
    )
    protocol.add_argument("--config", type=Path, help="dataset YAML to check for protocol drift")
    protocol.add_argument("--check", action="store_true", help="return nonzero on any drift")
    protocol.add_argument("--json", action="store_true", dest="as_json")
    protocol.set_defaults(handler=_cmd_protocol)

    plan = commands.add_parser("plan", help="write an explicit manifest for generation jobs")
    plan.add_argument("--config", type=Path, help="advanced generation YAML")
    plan.add_argument("--tier", choices=("tiny", "debug", "signal", "medium", "full"))
    plan.add_argument("--output", required=True, type=Path)
    _quality_profile_options(plan)
    _overrides(plan)
    plan.set_defaults(handler=_cmd_plan)

    generate = commands.add_parser("generate", help="generate deterministic, resumable data shards")
    generate.add_argument("--config", type=Path, help="advanced generation YAML")
    generate_destination = generate.add_mutually_exclusive_group()
    generate_destination.add_argument("--output", type=Path, help="exact output directory")
    generate_destination.add_argument(
        "--root", type=Path, help="parent directory; writes ROOT/pdeobs_TIER"
    )
    generate.add_argument("--tier", choices=("tiny", "debug", "signal", "medium", "full"))
    generate.add_argument("--plan", type=Path, help="JSONL plan produced by pdeobs plan")
    generate.add_argument("--array-index", type=int, help="run only this zero-based plan row")
    generate.add_argument(
        "--array-bundle-size",
        type=int,
        default=1,
        help="number of consecutive plan rows owned by one array element",
    )
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--num-workers", type=int, default=1, help="local shard processes")
    _quality_profile_options(generate)
    _overrides(generate)
    generate.set_defaults(handler=_cmd_generate)

    generate_case = commands.add_parser(
        "generate-case", help="generate one explicit PDE/boundary/setting/regime case"
    )
    generate_case.add_argument("--pde", required=True)
    generate_case.add_argument("--boundary", required=True)
    generate_case.add_argument("--setting", required=True)
    generate_case.add_argument("--param-regime", dest="param_regime", required=True)
    generate_case.add_argument("--num-samples", type=int, default=100)
    generate_case.add_argument("--root", type=Path, default=Path("data"))
    generate_case.add_argument("--resolution", type=int, default=128)
    generate_case.add_argument("--time-steps", type=int)
    generate_case.add_argument("--shard-size", type=int, default=100)
    generate_case.add_argument("--seed", type=int, default=20260804)
    generate_case.add_argument(
        "--tier", choices=("tiny", "debug", "signal", "medium", "full", "custom")
    )
    generate_case.add_argument("--force", action="store_true")
    generate_case.add_argument("--dry-run", action="store_true")
    _quality_profile_options(generate_case)
    generate_case.set_defaults(handler=_cmd_generate_case)

    download = commands.add_parser("download", help="download and verify a published dataset tier")
    download.add_argument(
        "--tier", required=True, choices=("tiny", "debug", "signal", "medium", "full")
    )
    download.add_argument(
        "--manifest",
        help="manifest URL/path; defaults to the publication-gated official release endpoint",
    )
    download_destination = download.add_mutually_exclusive_group()
    download_destination.add_argument("--output", type=Path, help="exact output directory")
    download_destination.add_argument(
        "--root", type=Path, help="parent directory; writes ROOT/pdeobs_TIER"
    )
    download.add_argument("--force", action="store_true")
    download.set_defaults(handler=_cmd_download)

    train = commands.add_parser("train", help="train a configured recovery or rollout baseline")
    train.add_argument("--config", type=Path, help="advanced experiment YAML")
    _experiment_selector(train)
    train.add_argument("--output", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--dry-run", action="store_true")
    _overrides(train)
    train.set_defaults(handler=_cmd_train)

    infer = commands.add_parser(
        "infer", help="run checkpoint inference and stream predictions to HDF5"
    )
    infer.add_argument("--config", type=Path, help="advanced experiment YAML")
    _experiment_selector(infer)
    infer.add_argument("--checkpoint", "--ckpt", required=True, type=Path)
    infer.add_argument("--output", type=Path, help="output .h5 or .hdf5 prediction file")
    infer.add_argument("--dry-run", action="store_true")
    _overrides(infer)
    infer.set_defaults(handler=_cmd_infer)

    evaluate = commands.add_parser("eval", help="evaluate a method or checkpoint")
    evaluate.add_argument("--config", type=Path, help="advanced experiment YAML")
    _experiment_selector(evaluate)
    evaluate.add_argument("--checkpoint", "--ckpt", type=Path)
    evaluate.add_argument("--pred", type=Path, help="prediction HDF5 produced by pdeobs infer")
    evaluate.add_argument("--metrics", help="comma-separated groups/names, such as rel_l2,spectral")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--dry-run", action="store_true")
    _overrides(evaluate)
    evaluate.set_defaults(handler=_cmd_eval)

    benchmark = commands.add_parser("benchmark", help="run a local configured method/split suite")
    benchmark.add_argument("--config", type=Path, help="advanced benchmark YAML")
    benchmark.add_argument("--preset", help="built-in benchmark preset")
    benchmark.add_argument(
        "--tier", choices=("tiny", "debug", "signal", "medium", "full"), default="medium"
    )
    benchmark.add_argument("--data", type=Path, help="override preset tier directory")
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--dry-run", action="store_true")
    _overrides(benchmark)
    benchmark.set_defaults(handler=_cmd_benchmark)

    aggregate = commands.add_parser(
        "aggregate", help="validate shards and aggregate result records"
    )
    aggregate.add_argument("--input", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--validate-shards", action="store_true")
    aggregate.add_argument("--skip-checksums", action="store_true")
    aggregate.add_argument(
        "--expected-plan",
        type=Path,
        help="strictly require the exact shards and row counts in a generation JSONL plan",
    )
    aggregate.add_argument("--group-by", default="method,task,split")
    _quality_gate_options(aggregate)
    aggregate.set_defaults(handler=_cmd_aggregate)

    quality = commands.add_parser(
        "quality", help="audit and aggregate stored PDE losses and dataset-quality checks"
    )
    quality.add_argument("--input", required=True, type=Path)
    quality.add_argument("--output", required=True, type=Path, help="quality-report JSON path")
    quality.add_argument(
        "--recompute",
        action="store_true",
        help="recompute each sample instead of trusting embedded quality metadata",
    )
    _quality_gate_options(quality)
    quality.set_defaults(handler=_cmd_quality)

    analyze = commands.add_parser(
        "analyze", help="summarize problem difficulty from JSON/CSV metric records"
    )
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--config", type=Path, help="optional difficulty-analysis YAML")
    analyze.add_argument("--primary-metric", help="metric used for worst-case rankings")
    analyze.add_argument("--top-k", type=int, help="number of ranked failures to retain")
    analyze.set_defaults(handler=_cmd_analyze)

    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import checks_succeeded, format_checks, run_doctor

    checks = run_doctor(cluster=args.cluster, require_gpu=args.gpu, offline=args.offline)
    print(format_checks(checks))
    return 0 if checks_succeeded(checks) else 1


def _component_names(*, discover: bool = False) -> dict[str, tuple[str, ...]]:
    # Importing component modules registers built-ins. Keep this explicit so a
    # fresh ``pdeobs list`` process does not depend on unrelated import order.
    from importlib import import_module

    for module_name in ("pdes", "settings", "masks", "metrics"):
        import_module(f"{__package__}.{module_name}")

    # Neural imports remain dependency-safe.
    from .methods import available_methods, discover_methods
    from .pdes import discover_generators
    from .registry import MASK_REGISTRY, METRIC_REGISTRY, PDE_REGISTRY, SETTING_REGISTRY

    if discover:
        discover_generators(on_error="warn")
        SETTING_REGISTRY.discover(on_error="warn")
        MASK_REGISTRY.discover(on_error="warn")
        METRIC_REGISTRY.discover(on_error="warn")
        discover_methods()
    return {
        "pdes": PDE_REGISTRY.names(),
        "settings": SETTING_REGISTRY.names(),
        "masks": MASK_REGISTRY.names(),
        "methods": available_methods(),
        "metrics": METRIC_REGISTRY.names(),
    }


def _cmd_list(args: argparse.Namespace) -> int:
    components = _component_names(discover=args.plugins)
    if args.kind != "all":
        components = {args.kind: components[args.kind]}
    if args.as_json:
        print(json.dumps(components, indent=2, sort_keys=True))
    else:
        for kind, names in components.items():
            print(f"{kind} ({len(names)}):")
            for name in names:
                print(f"  {name}")
    return 0


def _cmd_protocol(args: argparse.Namespace) -> int:
    from .config import load_config
    from .presets import default_generation_config
    from .protocol import protocol_report

    checked_config = None
    if args.config is not None:
        checked_config = load_config(args.config)
    elif args.check:
        checked_config = default_generation_config("full")
    report = protocol_report(checked_config)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        dataset = report["dataset"]
        print(report["title"])
        print(report["central_question"])
        print(
            f"Design: {dataset['macro_cases']} macro cases, "
            f"{dataset['regime_nodes']} regime nodes, {dataset['full_samples']} full samples"
        )
        print(
            f"Tasks: {len(report['tasks'])}; splits: {len(report['splits'])}; analyses: {len(report['analyses'])}"
        )
        gate = report["publication_gate"]
        print(
            "Official data release: "
            + (
                "published"
                if gate["official_release_published"]
                else "not published (validation-gated)"
            )
        )
        if "validation" in report:
            validation = report["validation"]
            print("Protocol check: " + ("PASS" if validation["valid"] else "FAIL"))
            for issue in validation["issues"]:
                print(f"  - {issue}")
    return 0 if report.get("validation", {}).get("valid", True) else 1


def _apply_quality_cli(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply explicit quality selectors after YAML/default resolution."""

    selected_profile = getattr(args, "quality_profile", None)
    selected_limit = getattr(args, "max_pde_loss", None)
    if selected_profile is None and selected_limit is None:
        return
    quality = dict(config.get("quality", {}))
    if selected_profile is not None:
        quality["profile"] = selected_profile
    if selected_limit is not None:
        thresholds = dict(quality.get("thresholds", {}))
        thresholds["pde_loss_normalized_max"] = selected_limit
        quality["thresholds"] = thresholds
    config["quality"] = quality


def _cmd_plan(args: argparse.Namespace) -> int:
    from .config import apply_overrides, expand_environment, load_config, save_resolved_config
    from .generation import write_generation_plan
    from .presets import default_generation_config
    from .provenance import collect_provenance, write_provenance

    config = (
        load_config(args.config, args.set)
        if args.config is not None
        else expand_environment(
            apply_overrides(default_generation_config(args.tier or "tiny"), args.set)
        )
    )
    if args.tier:
        config["tier"] = args.tier
    _apply_quality_cli(config, args)
    provenance = collect_provenance(config=config)
    generation_config = dict(config)
    generation_config["_provenance"] = provenance
    jobs = write_generation_plan(generation_config, args.output)
    save_resolved_config(config, args.output.with_suffix(".resolved.yaml"))
    write_provenance(args.output.with_suffix(".provenance.json"), config=config)
    print(f"Wrote {len(jobs)} generation jobs to {args.output}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .config import apply_overrides, expand_environment, load_config, save_resolved_config
    from .generation import run_generation
    from .presets import default_generation_config
    from .provenance import collect_provenance, write_provenance

    selected_tier = args.tier or "tiny"
    config = (
        load_config(args.config, args.set)
        if args.config is not None
        else expand_environment(apply_overrides(default_generation_config(selected_tier), args.set))
    )
    if args.tier:
        config["tier"] = args.tier
    _apply_quality_cli(config, args)
    selected_tier = str(config.get("tier", selected_tier))
    output = (
        args.output
        if args.output is not None
        else (args.root or Path("data")) / f"pdeobs_{selected_tier}"
    )
    generation_config = dict(config)
    generation_config["_provenance"] = collect_provenance(config=config)
    if args.array_index is None:
        context_dir = output / "_generation"
        save_resolved_config(config, context_dir / "resolved.yaml")
        write_provenance(context_dir / "provenance.json", config=config)
    result = run_generation(
        generation_config,
        output_root=output,
        plan_path=args.plan,
        array_index=args.array_index,
        array_bundle_size=args.array_bundle_size,
        force=args.force,
        dry_run=args.dry_run,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_generate_case(args: argparse.Namespace) -> int:
    from dataclasses import asdict, replace

    from .config import save_resolved_config
    from .generation import generate_job, jobs_from_spec
    from .provenance import collect_provenance, write_provenance
    from .schema import GenerationSpec, json_safe
    from .splits import TIER_SIZES

    inferred_tier = next(
        (name for name, size in TIER_SIZES.items() if int(size) == args.num_samples),
        "custom",
    )
    spec = GenerationSpec(
        pde=args.pde,
        boundary=args.boundary,
        setting=args.setting,
        regime=args.param_regime,
        num_samples=args.num_samples,
        resolution=args.resolution,
        seed=args.seed,
        time_steps=args.time_steps,
        shard_size=args.shard_size,
        tier=args.tier or inferred_tier,
        quality={
            "profile": args.quality_profile or "report",
            "thresholds": (
                {"pde_loss_normalized_max": args.max_pde_loss}
                if args.max_pde_loss is not None
                else {}
            ),
        },
    )
    output = args.root / "pdeobs_cases"
    jobs = jobs_from_spec(spec, output, include_tier_dir=False)
    canonical_case_id = "/".join((jobs[0].pde, jobs[0].boundary, jobs[0].setting, jobs[0].regime))
    resolved = {"generation_case": spec.to_dict(), "canonical_case_id": canonical_case_id}
    provenance = collect_provenance(config=resolved)
    jobs = [replace(job, provenance=provenance) for job in jobs]
    context_dir = output / "_generation" / Path(*canonical_case_id.split("/"))
    save_resolved_config(resolved, context_dir / "resolved.yaml")
    write_provenance(context_dir / "provenance.json", config=resolved)
    payload: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "complete",
        "case_id": canonical_case_id,
        "tier": spec.tier,
        "sample_count": spec.num_samples,
        "job_count": len(jobs),
        "output_root": str(output),
    }
    if args.dry_run:
        payload["jobs"] = [job.to_dict() for job in jobs]
    else:
        results = [generate_job(job, resume=not args.force, overwrite=args.force) for job in jobs]
        payload["results"] = [json_safe(asdict(result)) for result in results]
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from .download import DEFAULT_RELEASE_MANIFEST_URL, download_release

    output = (
        args.output
        if args.output is not None
        else (args.root or Path("data")) / f"pdeobs_{args.tier}"
    )
    manifest = args.manifest or DEFAULT_RELEASE_MANIFEST_URL
    paths = download_release(manifest, output, args.tier, force=args.force)
    print(f"Verified {len(paths)} files under {output}")
    return 0


def _selector_overrides(args: argparse.Namespace) -> list[str]:
    from .presets import normalize_mask, normalize_task

    overrides = list(args.set)
    if getattr(args, "model", None):
        raise ValueError("--model is used instead of --config; with YAML use --set method.name=...")
    values = {
        "task": normalize_task(args.task) if getattr(args, "task", None) else None,
        "data.root": str(args.data) if getattr(args, "data", None) is not None else None,
        "data.split": getattr(args, "split", None),
        "data.mask.protocol": (normalize_mask(args.mask) if getattr(args, "mask", None) else None),
        "data.filters.pde": getattr(args, "pde", None),
        "data.filters.boundary": getattr(args, "boundary", None),
        "data.filters.setting": getattr(args, "setting", None),
        "data.filters.regime": getattr(args, "param_regime", None),
    }
    overrides.extend(
        f"{key}={json.dumps(value)}" for key, value in values.items() if value is not None
    )
    return overrides


def _runner(action: str, args: argparse.Namespace) -> int:
    from . import runner
    from .presets import build_experiment_preset, normalize_task

    function: Callable[..., Any] = getattr(runner, action)
    config_free = args.config is None
    if config_free:
        missing = [
            name for name in ("task", "model", "data") if getattr(args, name, None) in {None, ""}
        ]
        if missing:
            raise ValueError(
                "without --config, the one-line interface requires "
                + ", ".join(f"--{name}" for name in missing)
            )
        factors = {
            "pde": getattr(args, "pde", None),
            "boundary": getattr(args, "boundary", None),
            "setting": getattr(args, "setting", None),
            "regime": getattr(args, "param_regime", None),
        }
        config_source: Any = build_experiment_preset(
            task=args.task,
            model=args.model,
            data=args.data,
            split=args.split or ("test" if action == "run_infer" else "iid"),
            mask=args.mask or "random_3pct",
            factors={key: value for key, value in factors.items() if value},
        )
        overrides: list[str] = list(args.set)
    else:
        config_source = args.config
        overrides = _selector_overrides(args)

    output = getattr(args, "output", None)
    if config_free and output is None and action == "run_train":
        output = Path("runs") / f"{args.model}_{normalize_task(args.task)}"
    if output is None and action == "run_infer":
        checkpoint = Path(args.checkpoint)
        run_root = (
            checkpoint.parent.parent
            if checkpoint.parent.name == "checkpoints"
            else checkpoint.parent
        )
        output = run_root / "preds.h5"
    result = function(
        config_path=config_source,
        overrides=overrides,
        output=output,
        checkpoint=getattr(args, "checkpoint", None),
        resume=getattr(args, "resume", None),
        dry_run=getattr(args, "dry_run", False),
    )
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    return _runner("run_train", args)


def _cmd_infer(args: argparse.Namespace) -> int:
    return _runner("run_infer", args)


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.pred is not None:
        if args.config is not None or args.checkpoint is not None:
            raise ValueError("--pred cannot be combined with --config or --checkpoint")
        from .evaluation import evaluate_prediction_file

        requested = (
            None
            if args.metrics is None
            else tuple(part.strip() for part in args.metrics.split(",") if part.strip())
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "prediction_file": str(args.pred),
                        "task": args.task or "sparse_recovery",
                        "metrics": requested,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = evaluate_prediction_file(
            args.pred,
            task=args.task or "sparse_recovery",
            metrics=requested,
            report_path=args.output,
            data_root=args.data,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    return _runner("run_evaluate", args)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from . import runner
    from .presets import build_benchmark_preset

    if args.config is None:
        if not args.preset:
            raise ValueError("benchmark requires --preset or --config")
        config_source: Any = build_benchmark_preset(
            args.preset,
            tier=args.tier,
            data=args.data,
        )
    else:
        if args.preset:
            raise ValueError("--preset and --config are mutually exclusive")
        config_source = args.config
    result = runner.run_benchmark(
        config_path=config_source,
        overrides=args.set,
        output=args.output,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from .aggregate import aggregate_path

    payload = aggregate_path(
        args.input,
        args.output,
        validate_shards=args.validate_shards or args.expected_plan is not None,
        verify_checksum=not args.skip_checksums,
        expected_plan=args.expected_plan,
        group_by=tuple(part.strip() for part in args.group_by.split(",") if part.strip()),
        quality_strict=args.quality_strict,
        max_pde_loss=args.max_pde_loss,
        require_all_pdes=args.require_all_pdes,
        require_validated_solvers=args.require_validated_solvers,
    )
    print(
        f"Found {payload['dataset']['shard_count']} shards and "
        f"{len(payload['leaderboard'])} aggregate result rows"
    )
    gate = payload.get("quality_gate", {})
    return 2 if isinstance(gate, dict) and gate.get("status") == "fail" else 0


def _cmd_quality(args: argparse.Namespace) -> int:
    from .quality import audit_dataset_quality, write_quality_csv
    from .storage import atomic_write_json

    report = audit_dataset_quality(
        args.input,
        recompute=args.recompute,
        strict=args.quality_strict,
        max_pde_loss=args.max_pde_loss,
        require_all_pdes=args.require_all_pdes,
        require_validated_solvers=args.require_validated_solvers,
    )
    atomic_write_json(args.output, report)
    csv_path = args.output.with_suffix(".csv")
    write_quality_csv(report, csv_path)
    gate = report["gate"]
    print(
        f"Audited {report['sample_count']} samples in {report['shard_count']} shards; "
        f"quality gate={gate['status']}; wrote {args.output} and {csv_path}"
    )
    return 2 if gate["status"] == "fail" else 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from .difficulty import analysis_output_paths, analyze_path, load_analysis_config

    config = load_analysis_config(args.config)
    if args.primary_metric is not None:
        config = replace(config, primary_metric=args.primary_metric)
    if args.top_k is not None:
        config = replace(config, top_k=args.top_k)
    report = analyze_path(args.input, args.output, config=config)
    destination = analysis_output_paths(args.output)["json"]
    print(
        f"Analyzed {report['record_count']} records with "
        f"{len(report['detected']['metrics'])} metrics; wrote {destination}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"pdeobs: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
