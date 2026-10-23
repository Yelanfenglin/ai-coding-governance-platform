from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "evaluation" / "regression.py"


def _load_checker(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("regression_check_under_test", CHECK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path
    return module


def test_missing_regression_inputs_fail_instead_of_ok(tmp_path: Path) -> None:
    checker = _load_checker(tmp_path)
    checker.RUNS = {"v2_dense@5": tmp_path / "missing.jsonl"}
    assert checker.main() == 1
    result = (tmp_path / "result.json").read_text(encoding="utf-8")
    assert '"status": "skip"' in result
    assert '"result": "FAIL"' in result
