"""Explore ADC API sequences against the source implementation on the host.

This supplements adc_migration's fixed contract, which remains responsible for
chip configuration and target integration. Completion is injected only while
ADS is set. No timing, analogue accuracy or target ABI is simulated here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.checks.adc_migration import host_files  # noqa: E402

OPERATIONS = ("start", "poll", "read", "read_null", "complete")
HARNESS = r'''
#include <stdio.h>
#include "zc.h"
#include "module.h"
void host_watchdog(void) {}
int main(void) {
 unsigned int op, sample, value;
 int result;
 ANSB=0x05; OEB=PUB=PDB=0xFF; GIEH=GIEL=1; T1CR=12; TC1EN=1;
 Demo_Adc0_Init();
 printf("-1 48879 %u\n", (unsigned int)ADS);
 while(scanf("%u %u", &op, &sample)==2) {
   result=-1; value=0xBEEF;
   switch(op) {
     case 0: result=Demo_Adc0_Start(); break;
     case 1: Demo_Adc0_Poll(); break;
     case 2: result=Demo_Adc0_Read(&value); break;
     case 3: result=Demo_Adc0_Read(0); break;
     case 4: if(ADS) { AD=(unsigned short)sample; ADS=0; } break;
     default: return 2;
   }
   printf("%d %u %u\n", result, value, (unsigned int)ADS);
 }
 return 0;
}
'''


def validate_sequence(value: object) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("sequence must be a nonempty list")
    for item in value:
        if not isinstance(item, dict) or item.get("op") not in OPERATIONS:
            raise ValueError(f"unknown operation: {item!r}")
        keys = {"op", "sample"} if item["op"] == "complete" else {"op"}
        if set(item) != keys:
            raise ValueError(f"unexpected operation fields: {item!r}")
        if "sample" in item and (type(item["sample"]) is not int
                                 or not 0 <= item["sample"] <= 65535):
            raise ValueError("complete.sample must be an integer in [0, 65535]")
    return value


def sequence_strategy(max_steps: int):
    """Reach each normal ADC state before exploring arbitrary API interleavings.

    Pure random lists spend most of a small search budget before any result is
    ready. These prefixes come from the source lifecycle: idle, converting,
    completed-but-unpolled, ready. Their operations remain in the counterexample.
    """
    from hypothesis import strategies as st

    samples = st.integers(min_value=0, max_value=65535)
    operations = st.one_of(
        st.sampled_from(OPERATIONS[:-1]).map(lambda op: {"op": op}),
        samples.map(lambda sample: {"op": "complete", "sample": sample}),
    )

    @st.composite
    def sequences(draw):
        state = draw(st.integers(min_value=0, max_value=min(3, max_steps - 1)))
        prefix = []
        if state >= 1:
            prefix.append({"op": "start"})
        if state >= 2:
            prefix.append({"op": "complete", "sample": draw(samples)})
        if state >= 3:
            prefix.append({"op": "poll"})
        return prefix + draw(st.lists(operations, min_size=1, max_size=max_steps - len(prefix)))

    return sequences()


class HostPair:
    def __init__(self, workspace: Path, out: Path):
        compiler = os.environ.get("CC") or "gcc"
        compiler = shutil.which(compiler) or compiler
        self.env = {**os.environ, "PATH": str(Path(compiler).resolve().parent)
                    + os.pathsep + os.environ.get("PATH", "")}
        self.executables: dict[str, Path] = {}
        header, registers = host_files()
        for side, relative in (
            ("source", "source/agent_demo/modules/demo_adc0"),
            ("target", "target/modules/demo_adc0"),
        ):
            build = out / side
            build.mkdir(parents=True, exist_ok=True)
            (build / "zc.h").write_text(header, encoding="utf-8")
            (build / "registers.c").write_text(registers, encoding="utf-8")
            (build / "module.h").write_text(
                f'#include "{(workspace / relative).as_posix()}.h"\n', encoding="utf-8",
            )
            (build / "sequence.c").write_text(HARNESS, encoding="utf-8")
            exe = build / "sequence.exe"
            command = [compiler, "-std=c11", "-fcommon", "-Wall", "-Werror", "-Wno-main",
                       "-I", str(build), "-I", str(workspace / side), "-x", "c",
                       str(workspace / (relative + ".c")), str(build / "registers.c"),
                       str(build / "sequence.c"), "-o", str(exe)]
            process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                     errors="replace", timeout=25, env=self.env)
            (build / "compile.json").write_text(json.dumps({
                "command": command, "exit_code": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr,
            }, indent=2), encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"{side} compilation failed: {process.stderr}")
            self.executables[side] = exe

    def trace(self, side: str, sequence: list[dict]) -> list[dict]:
        wire = "".join(f"{OPERATIONS.index(op['op'])} {op.get('sample', 0)}\n"
                       for op in sequence)
        process = subprocess.run([str(self.executables[side])], input=wire,
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", env=self.env, timeout=2)
        if process.returncode:
            raise RuntimeError(f"{side} execution exited {process.returncode}: {process.stderr}")
        rows = []
        for line in process.stdout.splitlines():
            result, output, ads = map(int, line.split())
            rows.append({"return": None if result == -1 else result, "output": output, "ADS": ads})
        if len(rows) != len(sequence) + 1:
            raise RuntimeError(f"{side} trace length: expected {len(sequence)+1}, got {len(rows)}")
        return rows

    def compare(self, sequence: list[dict]) -> dict:
        source, target = (self.trace(side, sequence) for side in ("source", "target"))
        first = next((i for i, (left, right) in enumerate(zip(source, target, strict=True))
                      if left != right), None)
        return {"source_trace": source, "target_trace": target,
                "first_difference": None if first is None else {
                    "step": first, "operation": {"op": "init"} if first == 0 else sequence[first-1],
                    "source": source[first], "target": target[first],
                }}


def check_sequences(workspace: Path, out: Path, *, max_examples: int = 300,
                    max_steps: int = 30, seed: int = 0,
                    replay: list[dict] | None = None) -> dict:
    """Return a bounded search result or replay a saved counterexample without searching."""
    workspace, out = workspace.resolve(), out.resolve()
    if out == workspace or workspace in out.parents:
        raise ValueError("sequence builds must be outside the checked workspace")
    if max_examples < 1 or max_steps < 1:
        raise ValueError("max_examples and max_steps must be positive")
    if replay is not None:
        validate_sequence(replay)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "passed": False, "status": "error", "validation_layer": "host_sequence_differential",
        "vendor_compile": "not_run", "hardware": "not_run",
        "settings": {"max_examples": max_examples, "max_steps": max_steps, "seed": seed},
        "search_strategy": "reachable_adc_states_v1",
        "mode": "replay" if replay is not None else "search", "evaluations": 0,
        "confirmation_evaluations": 0,
        "first_failing_sequence": None, "counterexample": None,
    }
    try:
        pair = HostPair(workspace, out)
        report["compile_seconds"] = time.perf_counter() - started

        def differs(sequence):
            report["evaluations"] += 1
            difference = pair.compare(sequence)["first_difference"]
            if difference is not None and report["first_failing_sequence"] is None:
                report["first_failing_sequence"] = sequence
            return difference is not None

        sequence = replay
        if replay is None:
            from hypothesis import Phase, find, settings
            from hypothesis import __version__ as hypothesis_version
            from hypothesis.errors import NoSuchExample

            report["hypothesis_version"] = hypothesis_version
            try:
                sequence = find(
                    sequence_strategy(max_steps), differs,
                    settings=settings(max_examples=max_examples, deadline=None, database=None,
                                      phases=(Phase.generate, Phase.shrink)),
                    random=random.Random(seed),
                )
            except NoSuchExample:
                sequence = None
        if sequence is not None:
            report["confirmation_evaluations"] += 1
            detail = pair.compare(sequence)
            if detail["first_difference"] is not None:
                counterexample = {"sequence": sequence, **detail}
                report["counterexample"] = counterexample
                path = out / "counterexample.json"
                path.write_text(json.dumps(counterexample, indent=2), encoding="utf-8")
                report["counterexample_path"] = str(path)
                report["replay_command"] = [
                    sys.executable, str(Path(__file__).resolve()), "--workspace", str(workspace),
                    "--out", str(out / "replay"), "--replay", str(path),
                ]
        report["passed"] = report["counterexample"] is None
        report["status"] = ("replay_matched" if replay is not None else "no_difference_found"
                            ) if report["passed"] else "difference_found"
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    report["elapsed_seconds"] = time.perf_counter() - started
    (out / "sequence_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def feedback(report: dict) -> dict:
    """Compact output suitable for verify_project's existing tool observation."""
    counterexample = report["counterexample"]
    return {"passed": report["passed"], "status": report["status"],
            "sequence": counterexample["sequence"] if counterexample else None,
            "first_difference": counterexample["first_difference"] if counterexample else None,
            "error": report.get("error"), "replay_command": report.get("replay_command")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    replay = None
    if args.replay:
        replay = json.loads(args.replay.read_text(encoding="utf-8"))["sequence"]
    report = check_sequences(args.workspace, args.out, max_examples=args.max_examples,
                             max_steps=args.max_steps, seed=args.seed, replay=replay)
    print(json.dumps(feedback(report), indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
