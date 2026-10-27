"""Manual evaluation entry point.

This is deliberately not CI: every subcommand is invoked by hand after a change
and reports a single exit code.  It only dispatches to the frozen-bank scripts;
it does not call any model, database, or paid API.

Usage::

    python -B evaluation/runner.py regression
    python -B evaluation/runner.py validate --pdf-root <pdf-root>
    python -B evaluation/runner.py score --run-dir <run> --split development
    python -B evaluation/runner.py --trace-path <trace.jsonl> status
    python -B evaluation/runner.py all
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "evaluation" / "retrieval"
REGRESSION = ROOT / "evaluation" / "regression.py"
SCORER = BANK / "score_results.py"
VALIDATOR = BANK / "validate_bank.py"
EVAL_RUNS = ROOT / "output" / "evaluation_runs"

ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "retrieval": tuple(
        ROOT / "output" / "retrieval_baseline_20261020" / name / "rules_v2_2" / "metrics.json"
        for name in ("v1_dense", "v2_dense", "v1_bm25", "v2_bm25")
    ),
    "answer": (
        ROOT / "output" / "answer_eval_v2_3_1_20261019" / "summary.json",
        ROOT / "output" / "answer_eval_v2_3_1_20261019" / "results.jsonl",
        ROOT / "output" / "answer_eval_v2_3_1_20261019" / "report.md",
    ),
    "ragas": (
        ROOT / "output" / "ragas_eval_v2_3_1_20261019" / "summary.json",
        ROOT / "output" / "ragas_eval_v2_3_1_20261019" / "results.jsonl",
        ROOT / "output" / "ragas_eval_v2_3_1_20261019" / "summary.md",
    ),
}


def _run(path: Path, argv: list[str]) -> int:
    if not path.is_file():
        print(f"missing script: {path}", file=sys.stderr)
        return 2
    previous = sys.argv
    try:
        sys.argv = [str(path), *argv]
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            print(code, file=sys.stderr)
            return 1
        return 0
    finally:
        sys.argv = previous


def regression() -> int:
    return _run(REGRESSION, [])


def validate(pdf_root: Path | None = None) -> int:
    argv = ["--pdf-root", str(pdf_root)] if pdf_root is not None else []
    return _run(VALIDATOR, argv)


def score(run_dir: Path, split: str, out_dir: Path | None = None) -> int:
    argv = ["--run-dir", str(run_dir), "--split", split]
    if out_dir is not None:
        argv.extend(["--out-dir", str(out_dir)])
    return _run(SCORER, argv)


def status(trace_path: Path | None = None) -> int:
    """Check that retrieval/answer/RAGAS artifacts exist and are non-empty.

    This is the no-paid, read-only half of the unified entry point: it never
    invokes a model and never treats a missing artifact as success.
    """

    missing: list[str] = []
    for group, paths in ARTIFACTS.items():
        absent = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
        print(f"{group:10s} {'ok' if not absent else 'missing'}")
        for path in absent:
            print(f"  missing/empty: {path}")
            missing.append(str(path))
    if trace_path is not None:
        if trace_path.is_file() and trace_path.stat().st_size > 0:
            print(f"{'trace':10s} ok ({trace_path})")
        else:
            print(f"{'trace':10s} missing ({trace_path})")
            missing.append(str(trace_path))
    print()
    print("RESULT:", "OK" if not missing else f"{len(missing)} artifact(s) missing")
    return 0 if not missing else 1


def _write_manifest(args: argparse.Namespace, exit_code: int, argv: list[str]) -> None:
    """Record the invocation next to its trace path for cross-artifact lookup."""

    trace_path = getattr(args, "trace_path", None)
    if trace_path is None:
        return
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = EVAL_RUNS / run_id
    try:
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "command": args.command,
            "argv": argv,
            "exit_code": exit_code,
            "trace_path": str(trace_path),
            "trace_exists": Path(trace_path).is_file(),
            "artifacts": {
                group: [str(path) for path in paths] for group, paths in ARTIFACTS.items()
            },
        }
        (out / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        print(f"manifest write failed: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-path",
        type=Path,
        help="optional JSONL trace from the same manual run (recorded in the manifest)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("regression", help="run frozen R050/R055/R056 + baseline drift checks")
    validate_parser = sub.add_parser("validate", help="validate the frozen bank")
    validate_parser.add_argument("--pdf-root", type=Path)
    score_parser = sub.add_parser("score", help="score an existing frozen retrieval run")
    score_parser.add_argument("--run-dir", type=Path, required=True)
    score_parser.add_argument(
        "--split", choices=("all", "development", "holdout"), default="development"
    )
    score_parser.add_argument("--out-dir", type=Path)
    all_parser = sub.add_parser("all", help="validate the bank, then run regression")
    all_parser.add_argument("--pdf-root", type=Path)
    sub.add_parser(
        "status", help="check retrieval/answer/RAGAS artifacts without paid calls"
    )

    args = parser.parse_args(argv)
    if args.command == "regression":
        code = regression()
    elif args.command == "validate":
        code = validate(args.pdf_root)
    elif args.command == "score":
        code = score(args.run_dir, args.split, args.out_dir)
    elif args.command == "status":
        code = status(args.trace_path)
    elif args.command == "all":
        code = validate(args.pdf_root)
        if code == 0:
            code = regression()
    else:
        code = 2
    _write_manifest(args, code, list(argv) if argv is not None else sys.argv[1:])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
