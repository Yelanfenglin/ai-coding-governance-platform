# Agent Runtime

为本地嵌入式开发提供模型调用、工具调度、会话、记忆、上下文管理与工程交付能力。Web 和 CLI 共用 `run_once` 装配入口与 `run_agent` 运行循环。

## 启动

按 [项目 README](../../README.md) 安装 `.[web,dev]` 并配置 `.env`，从仓库根目录 `npm run dev`。单独启动服务：

```powershell
$env:PYTHONNOUSERSITE='1'
.venv/Scripts/python.exe -m app.web --env-file .env --workspace <工程目录>
```

CLI 读取进程环境变量，Web 可通过 `--env-file` 加载 `.env`：

```powershell
.venv/Scripts/python.exe -m app.cli "核对当前 PWM 配置并按需求修改" --workspace <工程目录> --product M9P6362 --version V1.16 --session runs/demo.jsonl --json
```

默认模型 `deepseek-flash`，thinking enabled、reasoning effort high。`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL` 配置模型；`RAG_API_URL`、`RAG_API_KEY` 配置检索，其中 URL 包含 `/api/rag`。

## 工具与任务

- 通用工程工具：`list_files`、`read_file`、`write_file`、`edit_file`、`run_command`。
- 规格检索：`search_spec` 返回定位摘要，`read_context` 展开完整上下文；引用保留文档、片段、型号、版本、页码和章节。
- 模块工具：通过 `--demo-project` 或 `--firmware-project` 接入例程目录与模块生成；通用工作区仍可编辑和接入其他模块。
- 记忆工具：保存、替换、失效已确认事实，并检索历史会话；项目事实与用户画像分别存储。
- 子 Agent：`--subagents` 开放 `subagent`、`wait_agent`、`send_message`、`interrupt_agent`，用于只读研究和审阅。

`--max-steps`、`--max-seconds` 是可选任务预算；模型可见剩余量。预算耗尽、取消、失败与正常完成分别返回。实际模型 usage 来自 provider，不以字符估算费用。

## 状态与上下文

默认状态目录是所选工程的 `.agent-runtime/`。`--state-dir` 或 `AGENT_STATE_DIR` 可覆盖；事实与画像默认容量为 6000/3000 字符，通过 `--memory-chars`、`--profile-chars` 调整。

记忆带来源、适用范围与失效状态，每轮生成注入快照；历史对话通过 SQLite FTS5 按需检索。`--session` 续接消息与计量锚点，压缩检查点参与活动上下文重建。

上下文默认 1,000,000 tokens，用 `AGENT_CONTEXT_WINDOW` 或 `--context-window` 匹配模型配置。压缩保持工具调用配对、滚动合并摘要并重测压力；本地估算不替代服务端 token 校验。`--no-memory`、`--no-compaction` 可分别关闭。

## 工程检查与交付

宿主通过 `--verification <配置.json>` 提供固定检查命令：

```json
{
  "command": ["{python}", "tests/check_firmware.py"],
  "cwd": "{workspace}",
  "timeout": 60
}
```

`{python}` 为当前解释器，`{workspace}` 为选中工程，`{config_dir}` 为配置所在目录。模型调用 `verify_project` 获得实际输出并修复；完成后宿主对最终版本复验。检查器应将日志和构建产物写入 `.agent-runtime/` 或工程外。

`--delivery-dir` 可指定交付目录，保存 `delivery.json`、`changes.diff` 和报告。`record_implementation` 批量关联需求、实现理由、源码位置、规格证据与检查；`run_command`、`verify_project` 返回 `check_call_id` 供直接引用。

检查与最终源码版本不一致会标记过期；未关联检查仍显示未验证。Web 的“工程审阅”保留完整差异与未关联文件。任务在宿主检查和交付记录完成后结束。

| 退出码 | 含义 |
|---|---|
| 0 | 正常完成 |
| 1 | 运行失败 |
| 3 | 预算耗尽 |
| 4 | 工程清单检查失败 |
| 5 | 宿主独立检查失败 |

独立评分、固定演示与测试命令见 [开发与演示](../../docs/development.md)。本机文件和命令沿用当前用户权限；会话、密钥与运行缓存不提交。
