"""CLI 的默认记忆目录按测试隔离，不能读写开发者的项目记忆。"""

import pytest


@pytest.fixture(autouse=True)
def isolated_cli_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path / "agent-state"))
