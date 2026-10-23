# Agent Runtime 验收（2026-10-20）

本页保留首次验收记录；后续能力与真实任务结果见 [2026-10-23 验收摘要](docs/acceptance-20261023.md)。

## 结论

**条件通过。** 实现与 DSH 对齐版规划基本一致；验收中发现并修复了一处 `concludes_turn` 分组语义问题。当前 Python 版：**35 passed**，`ruff check .` 通过。

## 环境与命令

```powershell
cd apps/agent-runtime
.\.venv\Scripts\python.exe -m pytest -q --basetemp <tmp>
.\.venv\Scripts\python.exe -m ruff check .
```

- Python 3.13.14
- pytest 35 passed
- ruff All checks passed
- 测试全部使用 MockProvider / 注入 transport，不访问真实网络、不调用模型

## 已核对

### 终止语义

- 模型无 tool call → `completed`
- 工具结果 `concludes_turn=True` → `completed`
- provider `stop_reason="length"` → `max_tokens`
- 取消 → `aborted`
- provider 异常 / `stop_reason="error"` → `error`
- 无 StopPolicy / LoopGuard / max_steps 核心抽象

### 循环与工具

- turn 外层 / step 内层；
- 多个 tool calls 按模型顺序串行；
- 未知工具、缺参、工具异常 → observation，loop 继续；
- 取消时未派发 calls 补 synthetic `ABORTED_BEFORE_DISPATCH` 结果；
- 事件顺序：turn/step/assistant/tool/step_finished/turn_finished。

### 会话

- append-only JSONL 事件日志：`turn/start`、`step/start`、`message`、`step/end`、`turn/end`；
- 可重建 messages；
- `last_turn_open` 可识别中断 turn；
- 缺失文件按空日志处理。

### Provider

- MockProvider 脚本化、记录请求；
- OpenAiCompatibleProvider 可注入 transport；
- finish_reason 映射；
- malformed response / tool arguments 报错；
- 无真实网络测试。

## 验收中发现并修复

`_execute_tool_calls` 原先在第一个 `concludes_turn=True` 时直接返回，导致同一组后面 tool call 没有 result。  
DSH 的 `concludesTurn` 只标记本轮完成，同组其余 call 仍会执行并留下结果；已改为执行完整组、记录全部结果后再返回 `completed`，并补测试：

```python
async def test_concludes_turn_still_records_later_calls_in_the_same_group()
```

修复后：35 passed。

## 与 DSH 的有意简化

- 单 turn：`pending_inputs` 恒为空，没有 steering / follow-up；
- 串行工具，没有 parallel-safe / exclusive barrier；
- 无 session tree / branch / compaction；
- 无 `agent/turn-stopping` 扩展点；
- 无 retry / provider headers / 多 provider；
- 无 UI、MCP、多 Agent、沙箱。

这些都在 README 中登记，不能声称“完整复刻 DSH”。

## 其他注意

1. 普通 `pytest` 在本机可能因 Windows pytest 临时目录清理报 `PermissionError`，测试本身仍通过；验收用了显式 `--basetemp`。
2. `apps/agent-runtime/node_modules` 是 TS 版残留，已 gitignore；确认无人使用后可删。
3. `REACT_SYSTEM_PROMPT` 不会自动注入；调用方需要自行把 `system_message(react_system_prompt())` 放进 messages。
4. 规划文档还残留部分 TS 文案（§2/§5/§9），不影响代码，但交接前建议同步为 Python。
