# 2026-10-22：工程隔离、交付评分与子 Agent 选项的源码对应

代码和测试为事实源。本文件登记本次适配的参考、保留的行为与差异；不表示完整移植上游。

| 机制 | 上游函数与版本 | 本地落点 | 保留与适配 |
|---|---|---|---|
| 工作区归属 | Pi `cae9fd28a2d3d515c83ea495093e3768cc8efab6`，`packages/coding-agent/src/core/session-manager.ts::getDefaultSessionDirPath/getDefaultSessionDir` | `app/cli.py::run_once` | 以选中工程根决定默认状态目录；Pi 在用户目录编码 cwd，本地用工程内 `.agent-runtime`。显式目录继续覆盖，不自动搬迁旧数据。 |
| 新建子上下文 | DSH `aa8262ec091698bae9a6b04773a6b5b06ad4aef2`，`subagent-spawn-in-process::SpawnInProcessProvider.start`、`subagent-in-process-driver::startInProcessRun` | `app/subagents.py::SubagentRunner.execute` | 无父对话 seed，宿主 system 前缀加独立任务；子消息、会话与输出各自持有，复用 `run_agent`。 |
| 等待、失败、取消 | 同版本 `tool-subagent::settleForegroundRun/stopReasonError`、`subagent-in-process-driver::drivePublishedRun` | `SubagentRunner.execute`、原 loop 取消链 | 当前工具等待子运行结束；非 completed 标为错误 observation，部分输出保留。Python 用已有 Task/async generator 清理，没有另造调度 loop。 |
| 输出选择 | 同版本 `subagent::AssistantOutputFold/finalAssistantOutput` | `SubagentRunner.execute::observe` | 最后一条非空 assistant 正文；空正文不抹掉旧输出；无完整正文时保留文本流。当前 provider 的本地返回是文本，未移植上游多模态 ContentBlock 返回。 |
| 子用量归集 | OpenAI Agents SDK `841dfdd938bfd4ca052475eb76d6245a356868e0`，`src/agents/agent.py::Agent.as_tool` 的 nested ToolContext 共享 `usage` | `contracts.py::RunControl`、`loop.py` | 共用 usage 账本，根运行汇总，子记录保留自己的 usage；只记 provider 实际返回的用量，不推测未知费用。未引入 SDK 依赖。 |
| 父子运行预算 | 既有本地 `Budget` | `RunControl`、`loop.py::_run_steps/_budget_message` | 本地扩展，不声称复制 DSH。步骤在无 await 的区间内检查并领取；父子共用起始时间和步数。时间仍在步骤边界检查。 |
| 交付事实 | 本地既有 `cli.py::_project_files`、`trajectory_recorder`；对照 DSH `ui-deliverables::mutationPath` 的事实来自真实事件原则 | `app/delivery.py` | 复用快照和事件；版本对应、命令输出、原始检索结果来自宿主。diff 和评分是本地领域扩展，不声称 DSH 已有同款报告。 |
| 任务判分 | 本地 `apps/rag-api/evaluation/spec_retrieval_v2_2/score_results.py::main` | `evaluation/runner.py` | 冻结输入、逐项判断、失败保留在分母、JSONL/汇总输出；不调用模型。判据由 case 声明，其中 `memory_recall` 读取已保存的 response，不重跑模型。 |
| 工程行为 oracle | 既有 `output/autonomous_module_20261022/verify_firmware.py::main` | `evaluation/checks/firmware.py::main`、`evaluation/checks/multi_module.py::main` | 保留 C harness 的配置、接收、坏帧、发送、旧业务与工程设置检查；将机器路径与固定目录改为参数，并参数化 PWM duty register 与目标波特率。它是验收代码，不进入运行时生成逻辑。 |
| 宿主验收入口 | DSH `agent-loop/src/tool-calls.ts::commitReady/appendToolResult`，结果按调用顺序进入会话 | `project_verification.py::ProjectVerification.tool`、`cli.py::run_once`、`workspace_tools.py::execute_command` | 本地扩展：命令、工作目录和超时由宿主配置一次加载，模型不能通过工具参数替换 argv，只复用既有命令执行器与工具结果通道。开发中调用返回真实失败输出供下一步观察；交付阶段对最终版本重跑同一命令。未配置或预算先行耗尽记为 `unverified`；检查改动了工程文件记为 `source_error`，不判通过。 |

DSH 的 `foldConsumedWork` 处理执行过的任务如何终止，不是 token 账本；本次没有将两者混称。
Hermes 的 `get_memory_dir` 是 profile-scoped，不能将其误说成现有工程隔离的实现依据。

## 明确没有移植

首轮为 foreground one-shot SPAWN；后续选项扩展见下节。没有 Cordis 插件生命周期、
独立 durable Jobs 服务、递归、多模态 structured capture 或被杀进程中的任务恢复。
Codex `multi_agents.rs/spawn.rs` 仅用于交叉核对工具委派入口和父子关联，没有整套搬入。
CLI 的只读工具组合是首个宿主配置；`SubagentRunner` 接受工具/运行选项工厂。

## 后续：模型可选的委派能力

仍固定 DSH `aa8262ec091698bae9a6b04773a6b5b06ad4aef2`，没有把教程的五种拓扑硬编码成工作流。

| 上游函数 | 本地函数 | 保留语义与必要适配 |
|---|---|---|
| `subagent-fork-in-process::completedTurnPrefix/start` | `SubagentRunner._completed_messages/execute` | 只取最近一个已结束轮次之前的上下文，不取当前工具轮。DSH 复制源事件 seed；本地重放到该边界的活动消息，再写子日志，包含工具配对、reasoning 与压缩 checkpoint，宿主 child system 配置继续独立。 |
| `tool-subagent::resolveDelegationRun/settleForegroundRun` | `execute/_start/wait` | 前台等待、后台先返回 id；等待超时不取消子任务。统一工具中的 `context_mode` 和 `continuable` 参数是本地 API，DSH 原本通过 provider/config 装配这两项。 |
| `SubagentContinuationManager.sendMessage/deliverToChild` | `_message_tool/_start/_drive` | 运行中输入进入同一个 Inbox 的 next-step；空闲子会话用原日志开新激活。保留部分输出和异常终止状态，不把中断算完成。 |
| `createAgentMessage/createSettlementMessage` | `_message_tool/_drive` | 区分 Agent 自写消息与运行时完成通知，写入 `source` 并经 Inbox 注入；父级不会收到子级整条工具轨迹。模型返回只含结果、身份和必要诊断；完整审计保存在 result/session。 |
| `tool-subagent-control::interrupt_agent`、activation drain | `interrupt/close`、CLI finally | 中断当前激活，持续会话可再用；宿主退出时收拢所有活动任务。仅支持直接父子通信，无 peer/递归。 |
| 本地既有共享 Budget 与本轮真实八步耗尽案例 | `loop._budget_reason/_budget_message` | 本地适配：子任务可设置激活步数限额，默认保留一个父级答复步骤。同一次父运行内，共享步数与 usage 不因继续子会话而重置；历史 usage 不重复计入新的父运行。 |

后台任务不是独立 daemon。正常宿主关闭后，显式复用父 session 可按 id 重新加载持续子
会话；重放不重新 fork 新的父历史，也不重新执行旧工具。中断历史复用原 CLI 配对修复，
只补上下文中的中断说明，不伪造历史副作用。未认领队列仍为进程内数据；没有实现 DSH 的
完整冷恢复事务、durable Inbox、跨进程调度和并发父实例协调。

源码缓存与检查产物：`output/subagent_options_20261022/`。新反例在
`tests/test_subagent_modes.py`，默认 HTTP/SSE 接线在 `tests/test_subagents_http.py`。
这些是机制验收，不将前一次小任务对照结果改写成新选项已取得模型效果收益。

## 工程测试与命令反馈

工程测试命令有两种来源，都沿用 `run_command`、`ToolResult.is_error` 和同一个 loop：

1. 普通任务描述里的命令，不需要 CLI 参数。
2. 宿主用 `--verification` 指定的验收配置（`verify_project` 工具）：命令、工作目录、
   超时在启动时一次加载，模型不能通过工具参数替换命令。它在开发阶段返回真实退出码与输出，
   在交付阶段对最终版本重跑一次，避免“检查通过后又改了代码”仍显示通过。

参考仍为 DSH 固定提交的 `agent-loop/src/tool-calls.ts::commitReady/appendToolResult`：
命令结果按调用顺序进入会话，再供下一步骤观察。工程判据由工程或宿主提供。
没有额外自动重试循环，也没有单独的反思轮数；模型仍受当前 Budget 限制，是否满足任务要求
由独立评分判断。验收未运行时不写成通过：交付报告中的 `task_acceptance` 取
`passed`／`failed`／`unverified` 三态。
`evaluation.runner check` 复用原 `check_requirement`，在交付前执行 case 的全部判据并返回
组合结果，不需要先伪造一份 delivery。它是可选的本地工程评估入口，未声称上游已有同款机制。

任务评估的通过只覆盖其列出的 checks；人工介入未知时不判 autonomous pass，未执行的
厂商编译和硬件验证不升级成通过。已有案例不是独立盲测，离线 HTTP stub 不证明模型质量。

## 反例与回归入口

- `tests/test_cli.py::test_default_memory_and_history_belong_to_the_selected_workspace`：同型号
  的工程 B 不得召回 A 的时钟事实或历史；重开 A 仍能召回，显式路径覆盖继续有效。
- `tests/test_delivery_evaluation.py`：模型报完成但要求未满足、旧命令结果、交付后修改、缺失
  run 均不能获得任务通过；实际 C oracle 能检出 USART TX 映射的反例。
- `tests/test_host_verification.py`：宿主配置的验收命令跑真实子进程，覆盖三种关键情况——
  Agent 没调用检查就报完成、检查通过后又改坏文件、验收命令本身失败（不存在／超时／改动工程）。
  另加预算先行耗尽（记 `unverified` 且不跑交付前检查）与恢复会话时改用当前宿主配置。
  检查通过只证明“最终版本通过该命令”，不升级为厂商编译或硬件通过。
- `tests/test_project_verification.py`：失败输出进入模型输入、修复后重跑、恢复会话时检查属于
  当前轮输入。
- `tests/test_memory_recall_check.py`：`memory_recall` 判据只读已保存 response；缺失事实、
  数值错、布尔被整数冒充、带工具调用或非新会话均不判通过，未保存 response 时显式报错而非默认通过。
- `tests/test_subagents.py`：父历史不泄入、并行确实发生、共享预算不超领、用量不漏记或双记、
  两种父取消关闭流、CLI 的模型/thinking/effort 继承与只读工具装配。
- `tests/test_subagents_http.py`：默认 HTTP/SSE transport、子 RAG 工具、主 Agent 写文件与执行
  命令、会话和交付评分在同一次离线接线里跑通。

参考源码缓存及本次复跑产物在本地 `output/agent_delivery_20261022/`，未入库，原历史验收产物不覆盖。
宿主验收入口的复跑记录在 `output/verify_after_thread_20261023/`（`pytest-full.txt` 为全量结果），
`output/delivery_entry_20261022/` 保留改动前快照与定向 pytest 目录。

源码入口：

- [Pi session-manager.ts（固定 commit）](https://github.com/badlogic/pi-mono/blob/cae9fd28a2d3d515c83ea495093e3768cc8efab6/packages/coding-agent/src/core/session-manager.ts)
- [DSH foreground 工具（固定 commit）](https://github.com/deepseek-ai/deepseek-harness/blob/aa8262ec091698bae9a6b04773a6b5b06ad4aef2/packages/subagent/tool-subagent/src/index.ts)
- [DSH in-process driver（固定 commit）](https://github.com/deepseek-ai/deepseek-harness/blob/aa8262ec091698bae9a6b04773a6b5b06ad4aef2/packages/subagent/subagent-in-process-driver/src/index.ts)
- [Agents SDK Agent.as_tool（固定 commit）](https://github.com/openai/openai-agents-python/blob/841dfdd938bfd4ca052475eb76d6245a356868e0/src/agents/agent.py)
