"""Command-line interface for generation, training, evaluation, and SeaWulf."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import __version__

DEFAULT_RELEASE_MANIFEST = (
    "https://github.com/ru1ch3n/PartialObs--PDEBench/releases/latest/download/manifest.json"
)


def _overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted YAML key; repeat as needed",
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

    plan = commands.add_parser("plan", help="write an explicit manifest for generation jobs")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--tier", choices=("tiny", "debug", "signal", "medium", "full"))
    plan.add_argument("--output", required=True, type=Path)
    _overrides(plan)
    plan.set_defaults(handler=_cmd_plan)

    generate = commands.add_parser("generate", help="generate deterministic, resumable data shards")
    generate.add_argument("--config", required=True, type=Path)
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--tier", choices=("tiny", "debug", "signal", "medium", "full"))
    generate.add_argument("--plan", type=Path, help="JSONL plan produced by pdeobs plan")
    generate.add_argument("--array-index", type=int, help="run only this zero-based plan row")
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    _overrides(generate)
    generate.set_defaults(handler=_cmd_generate)

    download = commands.add_parser("download", help="download and verify a published dataset tier")
    download.add_argument(
        "--tier", required=True, choices=("tiny", "debug", "signal", "medium", "full")
    )
    download.add_argument("--manifest", default=DEFAULT_RELEASE_MANIFEST)
    download.add_argument("--output", type=Path, default=Path("datasets"))
    download.add_argument("--force", action="store_true")
    download.set_defaults(handler=_cmd_download)

    train = commands.add_parser("train", help="train a configured recovery or rollout baseline")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output", type=Path)
    train.add_argument("--resume", type=Path)
    _overrides(train)
    train.set_defaults(handler=_cmd_train)

    infer = commands.add_parser(
        "infer", help="run checkpoint inference and stream predictions to HDF5"
    )
    infer.add_argument("--config", required=True, type=Path)
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument(
        "--output", required=True, type=Path, help="output .h5 or .hdf5 prediction file"
    )
    _overrides(infer)
    infer.set_defaults(handler=_cmd_infer)

    evaluate = commands.add_parser("eval", help="evaluate a method or checkpoint")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--output", type=Path)
    _overrides(evaluate)
    evaluate.set_defaults(handler=_cmd_eval)

    benchmark = commands.add_parser("benchmark", help="run a local configured method/split suite")
    benchmark.add_argument("--config", required=True, type=Path)
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
    aggregate.set_defaults(handler=_cmd_aggregate)

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
    from .registry import MASK_REGISTRY, METRIC_REGISTRY, PDE_REGISTRY, SETTING_REGISTRY

    if discover:
        PDE_REGISTRY.discover(on_error="warn")
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


def _cmd_plan(args: argparse.Namespace) -> int:
    from .config import load_config, save_resolved_config
    from .generation import write_generation_plan
    from .provenance import collect_provenance, write_provenance

    config = load_config(args.config, args.set)
    if args.tier:
        config["tier"] = args.tier
    provenance = collect_provenance(config=config)
    generation_config = dict(config)
    generation_config["_provenance"] = provenance
    jobs = write_generation_plan(generation_config, args.output)
    save_resolved_config(config, args.output.with_suffix(".resolved.yaml"))
    write_provenance(args.output.with_suffix(".provenance.json"), config=config)
    print(f"Wrote {len(jobs)} generation jobs to {args.output}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .config import load_config, save_resolved_config
    from .generation import run_generation
    from .provenance import collect_provenance, write_provenance

    config = load_config(args.config, args.set)
    if args.tier:
        config["tier"] = args.tier
    generation_config = dict(config)
    generation_config["_provenance"] = collect_provenance(config=config)
    if args.array_index is None:
        context_dir = args.output / "_generation"
        save_resolved_config(config, context_dir / "resolved.yaml")
        write_provenance(context_dir / "provenance.json", config=config)
    result = run_generation(
        generation_config,
        output_root=args.output,
        plan_path=args.plan,
        array_index=args.array_index,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from .download import download_release

    paths = download_release(args.manifest, args.output, args.tier, force=args.force)
    print(f"Verified {len(paths)} files under {args.output}")
    return 0


def _runner(action: str, args: argparse.Namespace) -> int:
    from . import runner

    function: Callable[..., Any] = getattr(runner, action)
    result = function(
        config_path=args.config,
        overrides=args.set,
        output=getattr(args, "output", None),
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
    return _runner("run_evaluate", args)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    return _runner("run_benchmark", args)


def _cmd_aggregate(args: argparse.Namespace) -> int:
    from .aggregate import aggregate_path

    payload = aggregate_path(
        args.input,
        args.output,
        validate_shards=args.validate_shards or args.expected_plan is not None,
        verify_checksum=not args.skip_checksums,
        expected_plan=args.expected_plan,
        group_by=tuple(part.strip() for part in args.group_by.split(",") if part.strip()),
    )
    print(
        f"Found {payload['dataset']['shard_count']} shards and "
        f"{len(payload['leaderboard'])} aggregate result rows"
    )
    return 0


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
