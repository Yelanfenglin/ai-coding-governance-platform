"""Frozen PWM 40% / USART 57600 / ADC preservation checks, migrated from acceptance.

Targets are case requirements. The oracle executes delivered C, not its own copy
of the implementation. This is host simulation, not vendor/hardware validation.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.checks.adc import build_checker  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        code = build_checker()(
            workspace=args.workspace, seed=args.seed, out=args.out,
            duty_register=80, baudrate=57600, compiler=os.environ.get("CC", "gcc"),
        )
    report = json.loads((args.out / "independent_checks.json").read_text(encoding="utf-8"))
    issues = []
    if code == 0:
        state = json.loads((args.workspace / "demo-project.json").read_text(encoding="utf-8"))
        modules = {item["id"]: item for item in state["modules"]}
        observed = json.loads(report["checks"]["configuration_and_old_behavior"]["stdout"])["baud"]
        recorded = modules["usart0"].get("actual", {}).get("baudrate_bps")
        if not isinstance(recorded, (float, int)) or abs(recorded - observed) > 1:
            issues.append(f"USART0 actual.baudrate_bps: expected {observed}, actual={recorded!r}")
        if modules["usart0"].get("parameters", {}).get("baudrate") != 57600:
            issues.append("USART0 parameters.baudrate: expected 57600")
    print(json.dumps({"status": report["status"], "failures": report["failures"],
                      "manifest_issues": issues}, ensure_ascii=False, indent=2))
    return 1 if code or issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
