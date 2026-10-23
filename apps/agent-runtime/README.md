# agent-runtime

最小通用 Agent Runtime：DSH 风格的 **turn / step** loop + 工具执行 + 会话事件日志。
运行时保持模型与工具可替换；CLI 装配模型、HTTP RAG、本地编程工具、记忆与压缩。

当前验收范围与真实任务结果见 [2026-10-23 验收摘要](docs/acceptance-20261023.md)。

## CLI 入口

配置 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`，RAG 服务配置为
`RAG_API_URL=http://127.0.0.1:8000/api/rag`。例如：

DeepSeek 使用 `OPENAI_MODEL=deepseek-flash`（当前官方版本 V4.1-Flash-1017），
`OPENAI_BASE_URL=https://api.deepseek.com`。CLI 默认显式发送
`thinking={"type":"enabled"}`、`reasoning_effort="high"`；可通过 `--thinking`
与 `--reasoning-effort low/high/max` 或 `OPENAI_THINKING`、`OPENAI_REASONING_EFFORT`
调整。关闭 thinking 时不发送推理强度。压缩复用相同配置。
推理字段独立保存到会话并按原文回传，不拼进答案正文。

```powershell
$env:PYTHONNOUSERSITE='1'
.venv/Scripts/python.exe -m app.cli "M8P515 的 PWM 有几个通道？" --product M8P515 --version V2.3 --session runs/demo.jsonl --max-steps 8 --max-seconds 120
```

CLI 默认启用事实库、用户画像、历史会话检索与自动压缩。`--state-dir`（或
`AGENT_STATE_DIR`）指定持久目录；默认位于所选工作区的 `.agent-runtime/`，未选工作区时
位于当前目录。显式相对路径仍相对启动目录解析；旧目录不会自动迁移或混入新工程。
目录归属参照 Pi 的 cwd-scoped session directory，本地采用工程内目录而非 Pi 的用户目录编码。
两个记忆库默认预算
6000/3000 字符，可用 `--memory-chars` / `--profile-chars` 调整；记忆工具通过
`target=project/user` 选择库，型号/版本默认来自 CLI 参数。模型只保存明确确认的事实，
不把时钟频率推断为某种时钟源，不编造年份。纠正来源时 `replace_fact` 可同时更新
`source/scope/applies_to`；`source=""` 清除错误来源，省略字段则保留。

`search_spec`默认按每条900字符返回带引用的定位摘要；截断会明确标注。用`read_context`
展开选中的完整证据，或显式`max_chars=0`返回检索原文。检索顺序、ID和服务端排名不变。
编程指引要求新数值实际运行计算、改动后执行可用验证；其效果由任务验收判断，
不能仅凭这条指引声称模型推理正确。

工程已有可信测试时，在任务描述中给出测试命令，例如 `python tests/check_firmware.py`。
模型沿用 `run_command` 执行、读取退出码和输出、修复后复验；实际执行结果记入交付产物。
判据由工程/宿主提供，不从生成代码反推；模型的 completed 状态不能代替独立评分。

需要固定工程验收入口时，用户或宿主显式提供 `--verification <verification.json>`。
配置使用 argv，默认从所选工程目录执行，不经过 shell。例如：

```json
{
  "command": ["{python}", "tests/check_firmware.py"],
  "cwd": "{workspace}",
  "timeout": 60
}
```

`{python}` 使用当前项目解释器；`{workspace}` 是所选工程绝对路径；`{config_dir}` 是
验收配置所在目录。相对 `cwd` 以所选工程为基准，配置文件路径以启动目录为基准。
模型通过无参数工具 `verify_project` 执行它，失败输出进入原有工具反馈链路，供修复后复验。
模型报完成后，宿主再检查最终文件版本；不增加模型请求或额外的自动修复循环。
验收命令应把日志和构建产物写到 `.agent-runtime/` 或工程外，改动被跟踪的工程文件会失败。
这是本机命令执行，配置和检查器应由工程维护者提供；该入口不提供 OS 沙箱或检查器写保护。

安装项目后可用 `agent-runtime` 和 `agent-evaluate` 从其他目录启动，避免依赖仓库 cwd：

```powershell
python -m pip install -e ".[http]"
agent-runtime "实现本轮需求并验证" --workspace <工程目录> --verification <verification.json> --json
agent-evaluate check --case <case.json> --workspace <工程目录> --out <检查产物目录>
```

命令需在当前 Python 环境的 Scripts/PATH 中，也可用环境内可执行文件的完整路径。
启用 `--verification` 时，默认交付目录是工程状态目录下的 `deliveries/<run>/`，可用
`--delivery-dir` 覆盖。`verification.status` 和交付中的 `task_acceptance` 分别记录
`passed/failed/unverified`，通过的范围是已配置的检查。未配置验收或运行提前终止记为
`unverified`；最终检查失败退出码为 5，预算耗尽仍为 3，清单门禁失败仍为 4。

评测侧有冻结任务 case 时，可在开发中手动运行同一组行为、清单及保护项检查，
任一失败返回非零：

```text
python -m evaluation.runner check --case <case.json> --workspace <工程> --out <检查产物目录>
```

也可在任务描述中提供这条命令，让模型读取组合检查结果。`check` 是可选评测入口，
不评定运行是否完成；交付后的 `score` 仍核对最终版本、运行状态与答案。

主机固件检查位于 `evaluation/checks/firmware.py`（PWM/USART/TC1）和
`evaluation/checks/adc.py`（在相同检查上增加 ADC）。失败报告先列出失败项及预期值、
实际值，例如 `ADC.Init.ADS: expected=0 actual=1`，同时保留逐项 stdout/stderr。
它们检查规格约束和主机行为，不替代厂商工具链编译或板卡验证。

`--session` 续接消息与计量锚点；有 session 时，历史会话在运行前后写入检索索引。
记忆快照只在请求中注入，不写成历史消息。`--no-memory` / `--no-compaction` 可分别关闭。

CLI 默认使用 1,000,000 tokens 上下文窗口，主 Agent 和子 Agent 共用配置。
保留 `AGENT_CONTEXT_WINDOW` 环境变量及 `--context-window` 参数；环境配置使用
`AGENT_CONTEXT_WINDOW=1000000`，替换原来的小窗口值。CLI 沿用环境变量读取方式，
不自动加载 `.env` 文件。计量参照 DSH：完整请求估算
（含记忆与工具 schema）加真实 usage
锚点和有符号增量。压缩后重测，默认至多两次；仍超阈值时输出警告并继续本轮，这是
DSH pre-step 的语义。字符估算不等同于模型分词，不承诺服务端硬窗口永不溢出。
`--json` 输出本轮答案、usage、压缩前后压力与警告，便于查看真实结果。

## 交付记录与任务评分

CLI 可加 `--delivery-dir runs/task-001/delivery`，落盘 `delivery.json`、`changes.diff`
和 `report.md`。记录来自实际文件快照与工具事件，包括需求、模型与预算配置、源文件
指纹、实际命令和输出、检索证据及会话位置；不会把模型报完成直接记成任务通过。
命令执行时的源码与最终交付不同，会标为 `matches_delivery=false`。该字段表示版本对应，
不代表命令足以验证需求。文件范围为源码、配置和说明文本，不是整个磁盘或硬件状态。

评分入口沿用 RAG 的冻结输入、独立判分和批次输出方式，手动执行，不调用模型：

```powershell
.venv/Scripts/python.exe -B -m evaluation.runner score --case evaluation/cases/pwm_duty_40.json --delivery runs/pwm40/delivery/delivery.json --workspace runs/pwm40/project --out runs/pwm40/score
.venv/Scripts/python.exe -B -m evaluation.runner batch --runs runs/declared-runs.jsonl --out runs/scores
```

每条 batch JSONL 声明 `case/delivery/workspace`，路径相对该清单所在目录。
`human_interventions` 可选，未知为 null，不按零次处理。缺失输入、错误初始工程、交付后
改动、验证失败都计入失败，不能跳过后从分母消失。`json_value` 支持标量期望，`unchanged`
检查明确保护的文件，`manifest` 复用现有检查器，`command` 执行验收者声明的独立验证。

现有任务是 development 验收案例，种子工程位于 `evaluation/fixtures/`。先复制任务
`seed` 指向的目录到新的工作目录，再用 CLI 开发并记录交付；评分核对起点与最终版本。
USART 主机验证沿用已有独立 C harness，需要主机 GCC，可用 `CC` 指定编译器。冻结种子
保留原字节，不参与 Ruff 格式改写。主机模拟与清单通过均不代表厂商编译或板卡通过。
迁移后的新增模块、跨模块改参、只读分析、指代与纠正、新会话记忆案例及重放方法见
[工程交付评测说明](evaluation/protocols/engineering_delivery.md)。记忆案例的 `memory_recall`
读取已保存 response，仅用于 `score/batch`，不适用于尚无回答的开发期 `check`。

## 工具式子 Agent

`--subagents` 开放以下工具，由模型按任务决定是否委派、怎样拆分和组合：

```text
subagent(prompt, context_mode="spawn"|"fork", run_in_background=false,
         continuable=false, max_steps=<optional activation allowance>)
wait_agent(agent_id, timeout_seconds=30)
send_message(agent_id, message, max_steps=<optional idle-activation allowance>)
interrupt_agent(agent_id)
```

SPAWN 从新上下文开始；FORK 重放父日志最近一个 `turn/end` 之前的活动消息上下文，
排除当前进行中的轮次，保留工具配对和压缩检查点。子运行沿用宿主自己的 system 配置。
前台直接等待结果；后台先返回 child id，后续可等待，也会通过 Inbox 发送完成通知。
持续子会话允许运行中纠正和空闲后追问；只返回投递确认，答案由后续结果或消息取得。
持续子 Agent 可以给直接父级发送消息；来源标为 `agent-message`，完成通知标为
`subagent-settled`，不当成人类新指令。CLI 为子运行装配 RAG、工程读取、例程读取和
历史检索工具，主 Agent 负责修改与交付；宿主工具工厂不绑定业务角色。

父子共用步数与时间预算，模型可看到剩余额度；usage 汇总包含子运行和摘要的已返回用量。
模型可指定子任务本次激活的步数额度；启用总步数限额时，子运行默认给父级保留一个
答复步骤（宿主可配置）。预留与局部额度是本地预算适配，不是 DSH 的原有策略。
`steps` 为主 Agent 步数，`total_steps` 含委派模型步骤；摘要和重试沿用原来不计 step 的
语义。取消沿现有事件和任务链传递。子会话、终止状态、部分输出及父调用 id 保存在
状态目录的 `subagents/<id>/`，检索原始内容可在对应子会话核对。

默认不开启。后台与父级在同一进程内并发，CLI 返回前关闭未完成子任务，不提供常驻
daemon。提供同一个显式父 `--session` 时，可按已知 child id 续接已保存的持续子会话，
不重新执行历史工具；不支持并发 CLI 共同驱动同一个父会话。未认领的 Inbox 消息不跨
进程恢复，也不自动重启被杀进程中的任务。未提供 `--session` 时，委派模式会在状态
目录创建本次父日志，供本次 FORK 使用。子记录 usage 按子会话累计，父 response usage
只记本次运行新产生的工作。多层递归、同级互发消息和并行写工程尚未移植。
新选项的模型效果需要真实任务评估；HTTP stub 验收只证明接线与生命周期。
源码函数映射与必要适配见 [源码对照](docs/source-alignment-agent-delivery.md)。

## 自主开发本地工程

`--workspace` 选择任意本地项目目录，不要求特定芯片、模块或工程文件格式。
LLM 负责理解需求、查资料、选择方案、编写和修正代码；工具实际执行操作：

- `list_files` / `read_file`：查看工程结构、源码与运行环境。
- `edit_file`：唯一原文匹配后局部替换，返回diff，保留UTF-8/BOM/GB18030编码和换行。
- `write_file`：创建文件或完整重写为UTF-8。
- `run_command`：在项目目录执行argv命令，返回退出码与输出；`python`使用当前解释器。
- 配置`--examples-root`后，`list_official_examples` / `read_official_example`可动态查阅
  目录中的官方ZIP及包内源码，不限下述模板目录。

```powershell
.venv/Scripts/python.exe -m app.cli "根据规格书和例程修改PWM配置，保留现有业务，计算并验证误差" --workspace <local-project> --examples-root <official-examples> --product <chip> --version <spec-version> --session runs/development.jsonl --max-steps 10 --max-seconds 180
```

任务步骤由模型按工具返回决定，未编码“某类需求必走某条生成路径”。频率/时序计算可由
模型编写可复跑脚本并实际执行；脚本的硬件公式仍需对照规格书。现有生成模板是快捷工具，
不覆盖时可直接编写代码。`--demo-project` / `--firmware-project`也注册同一组通用工具。
生成文件经直接编辑后按定制代码继续维护，模板重建仍会拒绝覆盖已修改文件。

工具分工参考[Pi源码](https://github.com/badlogic/pi-mono/tree/cae9fd28a2d3d515c83ea495093e3768cc8efab6/packages/coding-agent/src/core/tools)。
本地采用单处精确替换、Python进程执行及Windows编码适配，未照搬Pi的全部实现。
文件工具相对选中目录；命令以当前本机用户权限执行，不提供操作系统沙箱。

## 本地模块 Demo

`--demo-project` 将模块生成工具接入当前 Agent。Agent 直接写入指定本地目录；无需独立
页面、生成服务、上传或下载。`--examples-root` 或 `DEMO_EXAMPLES_ROOT` 指向已有官方例程
目录，内部应包含 `M9P6362/M9P6362_Demo_V1.07.zip`。

```powershell
.venv/Scripts/python.exe -m app.cli "生成 C0 的 1kHz、30% PWM，再加 TC1 的 10ms 定时事件" --demo-project runs/module-demo --examples-root <official-examples> --session runs/module-demo.jsonl --max-steps 8 --max-seconds 120
```

后续沿用相同的工程与会话，例如“只把占空比改成40%，其它不变”。生成器保留未修改参数
和其它模块，在写入前检查定时器/逻辑端口冲突；发现已有人工修改时拒绝覆盖。

首版支持 **M9P6362 / 规格书 V1.16 / 官方例程 V1.07** 的 PWM0、TC0 定时事件、TC1
定时事件，沿用官方 HIRC 32MHz / 8T 配置。PWM0 与 TC0 定时事件共享 TC0，不能同时作为
独立模块生成。新型号/外设的自主开发可使用通用工具；列入已审查模板目录则需补充源码
适配与验收，不能把一次模型生成直接列为稳定支持。

文件包括模块 `.c/.h`、单一 `main.C`、系统初始化、共享中断分发、ZCW-ICE 工程与
`README.md` / `SOURCES.md`。已有工程通过挂接初始化、轮询和中断函数接入；不会自动改写
已有业务代码。`SOURCES.md` 逐项记录官方函数和必要改动，寄存器量化后的实际值写入清单。
生成器升级后旧文档与新渲染结果可能不同，文件保护会拒绝覆盖；本版不做跨版本工程迁移。

不需要模型时，可直接运行同一生成流程：

```powershell
.venv/Scripts/python.exe -m app.demo_cli generate --examples-root <official-examples> --project runs/module-demo --module pwm0 --parameters '{"frequency_hz":1000,"duty_percent":30,"channels":[0]}'
.venv/Scripts/python.exe -m app.demo_cli show --examples-root <official-examples> --project runs/module-demo
```

结构检查和主机 C 测试不等同于厂商工具链编译或板卡运行。工程依赖厂商 SDK 的 `zc.h`，
生成结果不附带替代 SDK。物理引脚需按实际封装核对。

### 在已有工程中增量接入

`--firmware-project` 选择已有工程目录或明确的 `.prj` 文件，与新建 Demo 的
`--demo-project` 二选一。Agent 的读取、搜索、差异预览和写入都针对这个本地工程：

```powershell
.venv/Scripts/python.exe -m app.cli "保留原TC1和业务代码，在C0接入1kHz、30%的PWM" --firmware-project <firmware-directory-or-prj> --examples-root <official-examples> --session runs/integration.jsonl --max-steps 8 --max-seconds 120
```

读取基于真实 `.prj` 文件清单及本地包含关系，返回寄存器引用位置、入口与已审查时钟
配置。新增模块放在 `agent_demo/`，只给原 main/循环/中断插入调用，并在原工程文件中
追加源文件条目。原系统初始化、业务函数和工程选项保留；改动前副本位于
`.agent-demo-backups/`。后续局部参数更新保留工程师在原业务文件中的修改。

`read_project_file` / `search_project` 核对源码；`preview_demo_integration` 只返回差异；
`generate_demo_module` 实际接入。只要分析时不会自动写入。也可不用模型预览：

```powershell
.venv/Scripts/python.exe -m app.demo_cli generate --existing --preview --project <firmware-directory-or-prj> --examples-root <official-examples> --module pwm0 --parameters '{"duty_percent":30}'
```

首版覆盖已审查的 M9P6362 ZCW-ICE 工程选项、直接寄存器引用、单一 main 和顶层
`while(1)` / `for(;;)` 循环。接入中断模块时，中断向量需与 main 同文件。它不执行完整 C
预处理、宏展开或间接调用分析。资源清单是引用证据，不能代替完整硬件连线和可用性确认；
有未确认条件时先核对。模块目录只表示生成工具覆盖范围，不能推断芯片没有其它外设。

已有工程增量接入可使用模板或通用工具；配置方案由LLM结合资料推导并执行计算验证。
跨型号适配仍需另行验证。

## 任务清单与自动续接

CLI 注册 `todo_write`、`get_goal`、`create_goal`、`update_goal`，来自 DSH
`aa8262ec091698bae9a6b04773a6b5b06ad4aef2` 的 `tool-todo`、`goal`、
`tool-goal`、`goal-round-driver`。模型可直接完成普通任务；`create_goal`
只用于需要跨回复自动推进的目标，不是完成开发任务的前置条件。

- `todo_write` 整表替换，状态为 `pending / in_progress / completed`；
  CLI 允许多个事项并行。清单写入会话事件，下一个 turn 开始时重置。
- Goal 以同一会话中的 `goal/change` 事件保存，更新必须携带精确 id/revision；
  只有真正进入上下文的自动轮次消耗 round 上限。普通人类追问不消耗它。
- 自动续接只在当前 turn 结束、没有其它待处理输入、目标 active/armed 时启动。
  `--max-goal-rounds` 是新目标未显式指定时的默认上限，默认 256，与上游配置相同；
  每个目标的实际 cap 以 `get_goal` 为准。`--max-steps`、`--max-seconds` 是整个
  运行的硬预算，不随 turn/round 重置。
- 新进程恢复目标时保持 disarmed。模型可在用户请求后恢复 active/blocked 目标；
  paused 目标需要用户显式通过 `--resume-goal` 恢复，再配合问题和 `--session` 继续。
- 自动轮次中 complete/blocked 后按上游原文注入收尾指令，不硬截断为无答案。
  模型自报 blocked 的轮次门槛为 3；直接人类输入有独立的操作权限。

这些机制管理任务状态和自动续接，不保证模型能正确编程或在给定步数内完成。
Python 适配保留原文提示词与策略；单 CLI 宿主没有移植 Cordis、RPC、插件热卸载或
多 Agent 权限图。第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## turn 与 step

- **turn**：处理一批用户输入。每个 turn 以 `turn/start` 开始、`turn/end` 结束。
- **step**：一次 model request + 一组 tool calls。一个 turn 可以有多个 step。

`RunOptions.inbox` 支持 next-step steering / next-turn follow-up；CLI 每次提交一个问题。

## 终止条件

终止由模型输出、工具结果与可选预算驱动：

| 条件 | 状态 |
|---|---|
| 模型返回没有 tool call | `completed` |
| 工具结果 `concludes_turn=True` | `completed` |
| provider `stop_reason="length"` | `max_tokens` |
| 取消（`cancel_event`） | `aborted` |
| provider 抛错 / `stop_reason="error"` | `error` |
| 可选 step / wall-time 预算用尽 | `budget_exceeded` |

`RunOptions.budget` 是本地适配；CLI 通过 `--max-steps` / `--max-seconds` 配置。

## 流式输出

**唯一入口**：`generate_stream(request) -> AsyncIterator[StreamChunk]`。对齐 DSH
`packages/llm/llm/src/index.ts` 的 `LlmAdapter.stream()` —— 上游注释原文是
"Stream one model call as raw chunks. **The only required method.**"

没有 `generate()`，也没有「provider 没实现流式就回退」的分支。这是刻意的：两条路径
意味着两套独立解析器，同一个语义（空响应、finish 映射、tool call 参数）必须在两处
各写一遍，漏一处就只在其中一条路径上暴露。

### chunk 种类

对齐 DSH `StreamChunk`（`packages/llm/llm/src/types.ts`）：

| chunk | 作用 |
|---|---|
| `BlockStart(index, block_type)` | 开一块。`block_type` ∈ `text` / `reasoning` / `tool-call` |
| `TextDelta(index, text)` | 文本增量 |
| `ReasoningDelta(index, text)` | 推理增量 |
| `ToolCallDelta(index, call_id, name, arguments_delta)` | 工具调用增量（参数是原始 JSON 片段） |
| `BlockEnd(index, block)` | 关块，携带**权威**的组装结果 |
| `UsageChunk(usage)` | token 记账 |
| `StreamFinished(reason)` | 终止。`reason.kind` ∈ `stop` / `tool-calls` / `max-tokens` / `aborted` / `error` |

`app/assembler.py::BlockAssembler` 是 chunk → block 的唯一算法，直接移植上游
`assembler.ts`：delta-only 协议容错、`block-end` 首次关闭为准、`max-tokens` 丢弃
tool-call、中断只保留非空 text/reasoning。

### 行为约定

- **终止是数据，不是异常**：正常结束、截断、取消、失败都走 `StreamFinished`。
  适配器内部可以抛错，但收尾时由 `_turn_or_raise` 把 `error` / `aborted` 还原成带
  稳定 `code` 的 `ProviderError`（对应 DSH `adapterFailureChunk` + `finishError`）——
  这样重试策略仍按 `EMPTY_RESPONSE` / `SERVER` 这些码判断。
- **`block-end` / `usage` / `finish` 全部延迟到 `[DONE]`**：`finish_reason` 之后
  常跟一块独立的 usage-only trailer，提前收尾会丢掉它。
- **没有 `[DONE]` 才是 `STREAM_CLOSED`**（上游 `sse.ts`）。**缺 `finish_reason`
  不是错误** —— 上游是 `pendingFinish ?? {kind: 'stop'}`。
- **明确 stop 却零个 block** → terminal `error` finish，`code=EMPTY_RESPONSE`（可重试）。
  `length` 与 reasoning-only 不算空。
- **tool call 参数是原始 JSON 串**（`ToolCallBlock.arguments`）。坏 JSON 不在解析层
  报错，保留原文交给工具层（上游 `parseArguments` 的 `catch { return raw }`），
  工具返回「参数不是 JSON 对象」的入参错误。
- 文本与推理增量立刻以 `AssistantTextDelta` / `AssistantReasoningDelta` 转给调用方；
  最终消息在收尾时一次性落盘。
- transport 产出的文本**可以是「行」也可以是「块」**：httpx `aiter_lines()` 已去掉
  行尾换行符，含 `\n` 的块也支持，跨块边界不丢行。对「每次 yield 一整行、不带行尾」
  的 transport 也兼容，但要**同时**满足两个条件才切行：下一块以 `data:` 开头，且残留
  本身已是完整可解析的 SSE 行 —— 否则 `metadata: PWM` 这类正文会被误切。
- 等待下一块时**与取消信号并行等待**：卡住的流不必等下一块到达才发现取消。
  外部 `task.cancel()` / `wait_for` 超时也会取消并 await provider 子任务、关闭流，再上抛。
- 中断语义照 DSH `agent.ts`：
  - **取消中断** → `BlockAssembler.interrupted_blocks()` 的结果作为
    `Message(interrupted=True)` 落进 messages/session（上游 `interruptedBlocks` + `interrupted: true`）；
  - **错误中断** → 不把半成品写成 assistant message，只报错（上游只留 attempt 记录）；
  - 中断丢弃 tool-call block —— 中断发生在派发之前，保留它就得伪造一个结果。

## 错误分类

所有 provider 失败都收敛成 `app.errors.ProviderError`，带稳定 `code`（**路由只看 code，
不解析 message**）与 `retryable`：

| code | 触发 | 可重试 |
|---|---|---|
| `RATE_LIMIT` | 429 / rate limit | ✅ |
| `SERVER` | 5xx | ✅ |
| `TIMEOUT` | 超时、408 | ✅ |
| `TRANSPORT` | 连接/流截断 | ✅ |
| `EMPTY_RESPONSE` | stop 但无内容 | ✅ |
| `AUTH` | 401 / 403 | ❌ |
| `INVALID_REQUEST` | 400 / 413 / 请求体过大 | ❌ |
| `CONTEXT_WINDOW_EXCEEDED` | 400 且命中上下文窗口措辞 | ❌ |
| `QUOTA` | 配额/余额耗尽 | ❌ |
| `STREAM_CLOSED` | 流无收尾事件 | ❌ |
| `ABORTED` | 取消 | ❌ |
| `HTTP_<status>` | 其他状态码 | ❌ |

默认可重试集合直接取自 DSH `retry-policy.ts` 的 `DEFAULT_RETRYABLE_CODES`。
状态码映射照抄 DSH `llm-deepseek/src/adapter.ts::httpErrorCode`，异常分类照
`llm-pi-ai/src/stream.ts::classifyPiAiError`。

## 工具调度：并行 + exclusive 屏障

照 DSH `tool-calls.ts` 的调度器：

- 每个工具声明 `execution_mode`（`"parallel"` / `"exclusive"`），**默认 exclusive**，
  未声明的工具不会被并行；
- 按模型顺序扫描：连续的 parallel-safe call 进入有界并发池（`max_parallel_tool_calls`，默认 10），
  exclusive call 是屏障 —— 它独占执行，并且前后都形成屏障；
- 派发前会重新读取模式，所以「parallel, parallel, exclusive, parallel」会自然切成
  三组：池 → 屏障 → 池；
- 结果一律**按模型顺序**写回（先完成的也要等前面的 slot 就绪）；
- 取消时：已开始的 call 跑完并把结果落盘，未派发的补 synthetic
  `Error: tool call aborted before dispatch`；
- **外层 `task.cancel()` 收尾**：先取消/等待本组工具子任务，再按模型顺序提交**已派发的
  全部槽位** —— 有结果的原样落盘，被取消的补 `Error: tool call was cancelled before it
  returned a result`。不会因为「靠前的调用被取消、slot 永远不出现」而丢掉后面已经跑完
  的结果，也不会为了补日志去重跑工具。turn 边界同时写一条 `turn/end(aborted)`，
  日志里不留永远开着的 turn；
- **提交完成后再推进槽位**：对齐 DSH `tool-calls.ts::commitReady`。本地 append 是异步的，
  `commit_slot` 把那次写入放进独立任务并 `shield` 起来（取消不会打断它），写入返回后才
  `committed += 1`；父运行取消时先收完这次写入再传播取消。写入还在排队或已经落盘时取消，
  都只留唯一的真实结果；
- **收这次写入有上限**：`COMMIT_GRACE_SECONDS`（默认 1s）。DSH 的 append 是同步的、不会
  把取消拖住；本地是异步的，sink 卡住（网络盘 / 满盘 / 被锁文件）时无上限等待会让
  `cancel()` 永不返回、外层 `wait_for` 超时失效。超上限就不再等：结果已经在上下文里，
  槽位照常推进，不会再补 synthetic —— 这次写入最终是否落盘归 SessionSink 的断尾恢复
  边界（S03，未移植）；
- **step 边界在 finally 收尾**：每个 `step/start` 已落盘的 step 都会写一次 `step/end`
  （取消/异常也一样），没开始的 step 不会凭空补 `step/end`；收尾失败不掩盖原始异常；
- `concludes_turn` 不打断同一 step 内剩余的 tool call（照 DSH），只保证不再请求模型。

## Token 记账 / Session Tree / Compaction（Phase E）

**记账**：每个 step 的 usage 写进 `StepEndEntry.usage`，`AgentRunResult.usage` 是累计值；
上下文计量采用 DSH token-meter 的 usage 锚点加有符号表面增量；消息、临时记忆、
工具 schema 均计入。基础估算与上游一样使用字符/4和结构开销，仍非精确分词。
`StepEndEntry.context_anchor` 保存基准，CLI 续接时恢复。

**Session Tree**：JSONL 每行是一个节点 `{id, parent_id, event}`，append-only、不删历史；
`JsonlSessionStore.leaf_id` 是当前分支末端，`ancestors(node_id)` 可回溯到根。

**重放两种读法**：`load_messages()` 返回**活动上下文**（遇到 `CompactionEntry` 会应用摘要，
等价于运行时 `result.messages`）；`load_raw_messages()` 返回未经压缩的完整历史（审计/调试）。
继续会话请用前者，并注意初始上下文需由调用方先写入会话。

**Compaction**（`RunOptions.compaction = CompactionPolicy(...)`）：

```python
CompactionPolicy(context_window=1_000_000, threshold_ratio=0.8, retain_ratio=0.16, summarizer=...)
```

- 比例与默认值取自 DSH `compaction-basic/config.ts`（0.8 / 0.16）；
- 只在 **step 边界**触发（DSH 挂在 `agent/pre-step`），不在请求中途做；
- 按**消息组**切：一条 assistant tool call 与它的全部 tool result 是一组，不会被拆散
  （不会出现「保留了 tool result 却没有对应 assistant call」的请求）；
- `split_region` 对齐 DSH `region.ts:132-146`：`retain_tokens` 是**最少**保留量 ——
  从尾部向前累加到 `>= retain_tokens` 才停，允许最后一组超额；不再把它当上限
  （旧写法在最后一组放不下时会一条原文都不留，与上游语义相反）。
  区域为空（`keepFromIdx <= firstIdx`）就不压缩；
- **固定 system prompt 不参与压缩**，摘要请求的前缀是「实际请求的固定前缀 + 区域消息」，
  `COMPACTION_INSTRUCTION` 只作为**最后一条 user 消息**追加（DSH `region.ts:528-546`）——
  这样辅助调用是上次请求的真实前缀，provider 的前缀缓存可复用；不会另插一个摘要专用 system。
  「实际请求的固定前缀」包含本 turn 注入的 **memory block**（开启记忆时前缀仍逐条相同）；
  记忆块只参与请求组装，不写进 `messages` / 会话历史；
- 旧的 checkpoint 不是受保护前缀，而是区域里的普通 **user** 消息，会**并入新摘要**（滚动合并）；
- 摘要结构沿用 DSH `summarizer.ts` 的八段（目标 / 技术概念 / 文件 / 错误与修复 / 待办 /
  当前工作 / 下一步 / 关键上下文含决策与证据定位）；
- 摘要请求**保留原始消息结构**（DSH 同款）：`tool_calls`（名称 + 参数）与 `tool_call_id`
  原样传给摘要模型，不展平成文本 —— 否则配置类事实（频率、占空比…）会永久丢失。
  `flatten=True` 只是给无法接受 tool 消息的 provider 用的文本兜底；
- **落地形态是 user checkpoint**：`CHECKPOINT_PREAMBLE + <compacted-summary>…</compacted-summary>`
  （DSH `summarizer.ts::frameSummary`）。摘要内容由模型产生，写成 system 会改变指令优先级；
- 两条结果验收（缺一条就回退、保持原上下文）：摘要**非空**
  （`summarizer.ts:167`）、framed checkpoint **比被替换区域更短**（`region.ts:398-406`）；
  摘要流以 `error` / `aborted` / `max-tokens` 终止同样算失败 —— 被截断的 checkpoint
  看起来「非空且更短」，但不能用来替换有效历史；
- 压缩结果以 `CompactionEntry` 追加到会话，**原始事件不删除**，重放 = 原 system head +
  user checkpoint + 保留区原文（`apply_compaction` 与 `compact_context` 产出形状一致）；
- **摘要调用的 usage 独立记账**：只要摘要请求真的发出去了，它的用量就计入 `usage`，
  与 checkpoint 是否被采用无关（被非空/缩短验收拒绝时同样计入；每次调用只记一次，
  不会累加上一次调用的 `last_usage`）；
- 取消 / 中断**向上传播**（按错误码的值判断：`ProviderError.code` 是字符串，
  不能用 `is` 比枚举成员）；只有普通失败才回退到未压缩上下文；
- **压缩失败回退**到未压缩上下文，不中断请求。

## 长期记忆（Phase F，参照 Hermes Agent）

DSH 里没有 memory 包，所以参照 **Hermes Agent**（`NousResearch/hermes-agent`）的有界记忆设计。
三层，各司其职：

| 层 | 实现 | 装什么 |
|---|---|---|
| L1 事实库 | `JsonMemoryStore(path, max_chars=6000)` | 环境事实、项目约定、术语映射、已确认事实 + 证据定位 |
| L2 用户画像 | `JsonMemoryStore(path, max_chars=3000, default_scope="user")` | 语言/格式偏好、要不要 citation、禁止改硬件配置 |
| L3 会话回溯 | `SessionSearch(db)` — SQLite FTS5 | 历史会话原文，**不占记忆预算** |

抄来的四条原则：

- **有界**：写不下抛 `MemoryFullError`，**绝不静默淘汰**旧条目（Hermes 同款行为）；`replace` 同样受预算约束；
  预算按**字符数**计（`len(str)`），不是字节也不是 token；
- **frozen snapshot**：每个 turn 开始时注入一次、插在 system prompt 之后，turn 内不刷新
  （保 prefix cache，也避免上下文抖动）。快照**不进入 `messages` / 会话历史**，只在组装请求上下文时
  插入 —— 把上一轮 `result.messages` 传给下一次 `run_agent` 不会残留已失效的旧快照；
- **并发写入不丢事实**：持久化是 append-only journal（每条写入追加一条操作），不是「读整个文件再覆盖」，
  Agent 工具与外部 API 同时写不会互相覆盖；提交时加**两层锁**：进程内按路径共享的
  `threading.Lock` + 跨进程的 OS 文件锁（`<journal>.lock`，Windows `msvcrt.locking` /
  POSIX `flock`）。只有进程内锁时，两个 CLI 进程会各自依据旧状态通过预算检查；
  读取也走同一把锁，避免读到并发追加写了一半的行；
- **预算在提交时校验**：journal 追加会在锁内重新读取状态、检查预算、再写入，所以并发写也不会把总量
  顶穿预算（`remember()` 里的提前检查只是快速失败）；超限一律抛 `MemoryFullError`，不静默淘汰；
- **旧格式会迁移**：上一版 `{"items": [...]}` 的整文件 JSON 会被识别并转成 journal，不会静默变成空库；
- **失效时间落盘**：`invalidate()` 的时间写进 journal，重放时不再重新生成（否则无法审计实际失效时间）；
- **Agent 自管理**：`memory_tools(store)` 给出 `remember_fact` / `replace_fact` / `forget_fact`
  （写操作默认 exclusive 屏障），`search_sessions_tool(search)` 只读、可并行；外部代码也能直接调；
- **只存 declarative 事实**：日志、临时路径、能重新查到的东西一律不存。

失效（两条都要）：

- `invalidate(key)` **软失效**：保留记录 + `invalidated_at`，可审计"这条被推翻过"；
- `applies_to={"chip": "..."}`：与当前上下文冲突的条目**自动不召回**（换型号/换版本不用手动清）。
  工具与外部 API 都支持：`remember_fact` 的 `applies_to` 参数会透传；不带适用范围的芯片事实
  会被当成全局事实（所有型号都召回），写芯片事实时应显式带上。

注入块示例：

```text
## Memory (durable facts)
Stable facts from earlier work. Verify before relying on them: tool evidence and citations always win.
[37% — 2,210/6,000 chars]
chip.target: PIC16F877A (scope=project, applies_to=chip=PIC16F877A, source=chunk-42, confidence=1.00)
```

**记忆不替代 citation**：块头写死 verify-before-use，回答仍以工具证据为准。
命中 0 条时完全不注入。

FTS5 说明：`ADCON0` / `低两位` 这类整词能命中；中文子串（`通道` → `通道选择位`）不行，
零命中时回退 `LIKE`。兜底与 FTS 语义一致：多个关键词各自做子串匹配再 AND，不是整句相邻匹配；
`%` / `_` 按字面量转义。

检索字段（对齐 Hermes `hermes_state_fts.py:40-65` / `session_search_tool.py:238-251`）：

- 进索引的**不只 `content`**：`tool_name` 与 `tool_calls`（工具名 + 参数 + call id）同为索引列，
  否则「哪个工具、什么参数」这类领域事实查不回来；
- **只有工具调用、没有正文**的 assistant 消息不会被跳过（它的正文是空的，事实在参数里）；
- 工具结果消息把 `tool_call_id` 记进索引列，按 call id 能同时捞回「调用」和「结果」两条；
- 命中结果带 `tool_name` / `tool_calls` / `tool_call_id`，`render()` 一并输出，便于判断这是哪次调用；
- 排序用 FTS5 的 `ORDER BY rank`（上游默认），不是按插入顺序取前 N 条；
- `LIKE` 兜底同样覆盖 `tool_name` / `tool_calls`；
- 索引是 JSONL 的派生数据：旧库缺少工具字段时重建表，但**先把旧行搬过去**（新列留空）——
  查询端不会因为升级而静默清空历史；重新 `index_*` 某会话时才补全它的工具字段。

## 参照与改写边界（务必如实表述）

主体机制有开源参照，但**是 Python 重新实现，不是复用上游 SDK**，也不与上游完整等价：

| 机制 | 参照 | 本地差异（如实登记） |
|---|---|---|
| turn/step 循环、工具调度 | DSH `agent.ts` / `tool-calls.ts` | 是简化移植，未移植 prepare/finalize 插件阶段 |
| retry / steering | DSH retry-policy / inbox | 有界子集；Inbox 是进程内队列，不是持久队列 |
| 压缩 | DSH `compaction-basic` / `token-meter` | 保留量、前缀、checkpoint 与结果验收已对齐；计量含 usage 锚点与增减量，默认最多两次压缩后复查。仍使用估算，超阈值警告后继续；真实 overflow 恢复未移植 |
| 记忆 | Hermes `memory_tool_store` | 上游冻结到 session，本地**每个 turn 重建快照**；journal / 结构化元数据 / `applies_to` 是本项目改造；跨进程锁已对齐（OS 文件锁） |
| 会话检索 | Hermes `session_search_tool` | 本地独立 FTS5 索引 + 中文 LIKE 兜底；索引列（`content`/`tool_name`/`tool_calls`）与 `ORDER BY rank` 已对齐；差异：上游 CJK 视图排除 `role='tool'` 行，本地把工具结果正文也索引（用例依赖「工具返回里的事实可检索」），且无 lineage 去重 / 邻居窗口展开 |
| 预算 / 轨迹 / RAG 工具 / CLI | **本项目需求** | 不是 DSH/Hermes 原样实现（被核验版本 DSH 无内置 turn budget） |

不能声称"接入 Pi SDK"或继承上游正确性保证；上游源码只作为设计依据与行为对齐目标。

### 逐项交代：上游保证 → 本地保留/改变 → 反例

| 机制 | 上游保证 | 本地 | 证明反例（tests/） |
|---|---|---|---|
| 模型调用入口 | `LlmAdapter.stream()` 是**唯一**必需方法，没有 `generate()` / `complete()` | 保留：`generate_stream` 唯一入口，没有回退分支 | `test_streaming.py::test_provider_without_stream_is_rejected` |
| 流 chunk 协议 | block 生命周期（`block-start` / 带 `index` 的 delta / `block-end`）+ 独立 `usage` / `finish` chunk | 保留：`app/assembler.py` 逐条移植 `assembler.ts` | `test_providers.py::test_chunks_from_sse_opens_closes_blocks_and_defers_termination`、`…::test_sse_reasoning_opens_its_own_block_before_text` |
| 终止失败 | adapter 抛错被规范成 terminal `error` / `aborted` finish chunk | 保留：收尾再还原成带稳定 `code` 的 `ProviderError`，重试按码判断 | `test_semantic_regressions.py::test_empty_response_error_becomes_error_result` |
| 缺 finish_reason | `pendingFinish ?? {kind: 'stop'}` | 保留：默认 `stop`；只有**没有 `[DONE]`** 才是 `STREAM_CLOSED` | `test_providers.py::test_missing_finish_reason_defaults_to_stop`、`…::test_sse_without_done_is_stream_closed` |
| 工具参数坏 JSON | 保留原始文本交给工具 | 保留：`ToolCallBlock.arguments` 是原始串，工具层报入参错误 | `test_providers.py::test_tool_arguments_that_are_not_json_are_kept_raw` |
| 工具调度 | 按模型顺序提交；取消时先处理已启动调用再补未派发结果 | 保留；**新增**父 `task.cancel()` 时收拢工具子任务；提交完成后才推进槽位，取消先收这次写入（有 1s 上限，见 `COMMIT_GRACE_SECONDS`） | `test_semantic_regressions.py::test_tool_tasks_do_not_survive_parent_cancellation`、`test_parallel_tools.py::test_cancel_during_result_commit_leaves_one_result_and_closes_the_step`、`…::test_cancel_is_bounded_when_a_result_write_is_stuck` |
| step/turn 边界 | `step/end` 与 `turn/end` 都有 finally 收尾 | 保留：每个已开始的 step 关一次；单测覆盖「没开始就取消不补 `step/end`」 | `test_loop.py::test_cancel_before_the_first_step_writes_no_step_events` |
| 压缩触发 | 对当前 session 重新测量压力 | 同配置用 usage 锚点 + 有符号表面变化；计入记忆与工具 schema；配置变化重新估算，CLI 续接恢复锚点 | `test_context_pressure.py` |
| 压缩配对 | 不拆散 tool call / result | 保留（按消息组切） | `test_acceptance_regressions.py::test_compaction_preserves_tool_call_result_pairs` |
| 摘要前缀 | 复用有效请求前缀（system + tools + 前导消息） | 保留：`request_prefix` = system head + memory block，末尾只追加指令 | `test_compaction.py::test_summarization_request_reuses_the_memory_prefixed_request` |
| 摘要失败控制流 | 终止映射为 fail-closed 错误 | 保留；按 code **值**分流（`ABORTED` 上抛，其余回退） | `test_compaction.py::test_summary_abort_propagates_instead_of_falling_back` |
| 摘要用量 | 辅助调用 usage 单独记录 | 保留：每次实际调用记一次，与是否采用 checkpoint 无关 | `test_compaction.py::test_rejected_summary_still_records_its_usage`、`…::test_rejected_summary_usage_reaches_the_run_total` |
| 工具参数校验 | handler 之前校验支持的 JSON Schema；不支持/位置不对的关键字直接拒绝 | 保留有界子集；`integer` 按**数值**判（`1.0` 合法）、`const`/`enum` 按 JSON 值判（`true` ≠ `1`）；无效 `pattern`、未声明 `required`、子集外关键字都明确拒绝。其中数值/长度/`pattern` 约束是**本地扩展**（上游子集没有断言类约束），正确性由本项目用例保证 | `test_tool_schema.py` |
| 摘要输入 | 原始消息结构 + **tools** + `purpose` / `maxTokens` / `sessionId` | 保留消息结构、tools、purpose 与默认8192输出上限；sessionId 传输元数据未移植 | `test_compaction.py` / `test_providers.py` |
| 摘要取消 | agent signal 传入摘要请求 | 保留：`cancelled` 探针 + 竞争等待（10ms 轮询） | `test_semantic_regressions.py::test_summarization_is_cancellable` |
| 空响应 | stop 且零 block → `EMPTY_RESPONSE`（可重试） | 保留：single-path 判空；reasoning-only / length 不算空 | `…::test_empty_completion_is_retried_not_completed`、`test_reasoning_only_completion_is_not_empty` |
| 记忆有界 | 满时显式失败、不静默淘汰 | 保留；预算在**提交时**校验 | `test_reacceptance_regressions.py::test_concurrent_public_writes_cannot_exceed_budget` |
| 记忆并发 | 基于**目标路径**的锁 + 重读更新 | 路径共享锁 + OS 文件锁，锁内重读并提交；覆盖同进程与跨进程 | `test_semantic_regressions.py`、`test_memory.py` |
| 记忆快照 | 冻结到 session | **改变**：每个 turn 重建（新事实下轮可见） | `test_memory.py::test_memory_snapshot_is_replaced_not_stacked` |
| 保留量 | `retainRatio` = 最少保留尾部 | 保留：按完整消息组累加到最少保留量，最后一组可超额 | `test_compaction.py::test_split_region_keeps_the_tail_as_whole_groups` |

## Trajectory 评测（Phase G）

`trajectory_recorder()` 把事件流压成可断言的轨迹摘要，评测不只看最终答案：

```python
trajectory, on_event = trajectory_recorder()
result = await run_agent(RunOptions(..., on_event=on_event))
trajectory.finish(result)
trajectory.summary()
# {'status': 'completed', 'turns': 1, 'steps': 3, 'tool_calls': ['echo', 'echo'],
#  'retries': 1, 'compactions': 0, 'memory_recalls': 0, 'errors': [], 'total_tokens': 55}
```

记录：turn/step 数、工具调用顺序与是否报错、retry 次数、压缩次数、记忆召回次数、
被取消截断的 assistant 消息、终止原因、usage。

`tests/test_trajectories.py` 用**会校验 tool-call/result 配对的 strict provider 替身**
跑固定脚本，覆盖：多步工具、parallel/exclusive 屏障、retry、两种取消入口、
压缩后继续（重放 == 运行上下文）、记忆范围与预算。全部离线，不调用真实模型。

配对校验规则：每条 tool 消息必须对应到前面某条 assistant 的 tool_call；
**只有请求最后一条消息发起的调用**允许还没有结果（正要去执行）—— 一旦后面又出现别的消息，
未回答的旧调用就是结构错误。

## RAG 工具（`apps/rag-api`）

`app/rag_tools.py` 把检索服务接成两个工具（只读 → `parallel`，可与其他只读工具并发）：

| 工具 | 端点 |
|---|---|
| `search_spec` | `GET /search?question=&product=&version=&kind=&top_k=` |
| `read_context` | `GET /documents/{doc_id}/context/{chunk_id}` |

- 鉴权：`X-API-Key` 头（`RagClientConfig.api_key`）；
- **引用字段一律保留**：`document_id` / `chunk_id` / `product`（型号）/ `version` /
  `page_start-page_end`（页码）/ `section_path`，渲染成
  `[1] document=... chunk=... product=... version=... pages=12-13 section=...`。
  `document_id` 必须出现 —— `read_context` 需要 `document_id` + `chunk_id` 两个参数，
  而模型只能从这段文本里拿到它们；
- 契约对齐真实服务：版本形如 `Vx.y`，`kind` 属于 `TEXT/TABLE/VISUAL/MIXED/CODE`
  （客户端不做本地校验，只原样渲染）；
- 服务错误（4xx/5xx）变成 `ToolResult(is_error=True)` 交给模型处理，不崩 run；
- httpx 是可选依赖：未安装时报错明确；测试注入 `RagTransport` 离线跑。

```python
client = RagClient(RagClientConfig(base_url="http://127.0.0.1:8000/api/rag", api_key=...))
registry = ToolRegistry()
for tool in rag_tools(client, default_product="PIC16F877A", default_version="v1"):
    registry.register(tool)
```

## 最小 CLI

```powershell
$env:OPENAI_BASE_URL='https://api.deepseek.com/v1'; $env:OPENAI_API_KEY='...'
$env:RAG_API_URL='http://127.0.0.1:8000/api/rag'; $env:RAG_API_KEY='...'
python -m app.cli "M8P515 的 PWM 周期怎么算？" --product M8P515 --version V1.0
python -m app.cli "..." --session runs/session.jsonl --json          # 机器可读输出
python -m app.cli "..." --max-steps 8 --max-seconds 60               # 受控联调
```

- `--json` 输出 `status / turns / steps / usage / tool_calls / answer / resumed_messages / error`；
  `answer` **只取本轮 assistant 输出**（provider 在生成前失败时为空串，不会拿用户问题
  或续接进来的上一轮答案充数）；
- `--session` 会**续接**这个会话：先 `load_messages()` 取回历史，再把本轮问题接在后面，
  最后追加写入同一文件。续接时沿用会话自己的 system head，不重复插第二条 ——
  否则同一会话会出现两条 system、旧的固定前缀被替换；
- 续接前会**补齐工具调用配对**：中断留下的「只有调用没有结果」的历史补一条
  `Error: tool call was interrupted before it returned a result`（不重跑工具），
  配不上任何调用的孤立结果丢弃 —— 半截历史不能原样送下一轮；
- `--no-history` 保留旧的「只追加、不加载」行为（历史仍在文件里，只是这次不带入上下文）；
- 退出码：`0` 完成、`1` 失败、`3` 预算用尽。

## 运行预算（与取消分开）

```python
RunOptions(budget=Budget(max_steps=8, max_seconds=60))
```

- 步数或墙钟时间先到先停 → 状态 **`budget_exceeded`**（不是 `aborted`：
  配额用尽是正常终止，用户中断才是取消）；
- 停止时**保留已发生的工具结果与 usage**，`error` 里给出具体限额原因
  （`step budget exhausted: max_steps=8` / `time budget exhausted: max_seconds=60`）；
- 检查发生在 step 边界，不在请求中途做；
- 计时钟从 **run 开始**（跨 turn 累计，不随 turn 重置），步数同样跨 turn 累计。

启用预算时，每步主模型请求末尾附带当前 `<run_budget>`：配置的总步数、已用步数和
剩余步数，以及时间上限、已用时间和剩余时间快照。剩余步数包含本次回答；提示模型
预留交付机会，最后一步据已确认的证据总结，明确未完成项。是否完成仍由实际验收判断。
未配置的限额不显示；未启用预算时不注入。快照只属于当前请求，不写入会话或摘要，
不改变固定 system 前缀；上下文压力估算包含它。这是本地 Budget 的扩展，参考 DSH
在 step 边界注入运行上下文的方式，未声称是 DSH 原有预算机制。

## Steering / Follow-up（Inbox）

`RunOptions.inbox = Inbox()` 启用。两个队列，语义照 DSH `inbox.ts` + `agent.ts`：

| 调用 | 队列 | 注入时机 |
|---|---|---|
| `inbox.steer(msg)` | `next-step` | 当前 assistant turn 及其 tool calls 完成后注入，**不开新 turn** |
| `inbox.followup(msg)` | `next-turn` | 当前 turn 完全结束后开一个新 turn |
| `inbox.send(msg, target)` | 显式指定 | 必须选目标，不猜 |

- 每个 turn 的第一个 step 认领 `next-turn`，之后的 step 只认领 `next-step`；
- DSH 行为：turn 已结束但 `next-step` 还有输入时，steering 会继续占用**同一个 turn**；
- 取走的输入以 user message 落盘并发 `InboxClaimed` 事件；
- 取消时默认丢弃队列（DSH `cancel()` 默认清 inbox），`clear_inbox_on_cancel=False` 可保留；
- 没有 inbox 时行为与之前完全一致（单 turn）。

## Retry（只针对模型请求）

`RunOptions.retry_policy = RetryPolicy(...)` 开启；默认 `None`（不重试）。

```python
RetryPolicy(max_retries=5, base_delay=0.5, max_delay=10.0, jitter=True)
```

默认值直接取自 DSH `retry-policy.ts`（`maxRetries=5`、`initialDelayMs=500`、
`maxDelayMs=10000`、`jitterRatio=0.1`）。

- `max_retries` 是**首次请求之外**还能重试几次：5 表示最多 6 次请求；
- 只对 `ProviderError.retryable` 且 code 在 `retryable_codes` 内的失败重试；
- 退避照 DSH `localDelay`：`min(base * 2 ** (retry - 1), max_delay)` × 对称 jitter `[1-r, 1+r]`；
- provider 给的 `Retry-After` 优先；**超过 `max_delay` 时直接放弃重试**（DSH normal 模式行为）；
- 等待可取消：取消立即停止，不再发起下一次 attempt；
- 重试计数是 step 内的局部变量，每个 step 重新开始（DSH 的 retry projection 在 `step/start` 清空）；
- **已经有可见输出的流式请求不重试**（重试会让调用方看到两份内容）；
- 每次重试发 `ProviderRetryScheduled` / `ProviderRetryStarted` 两个事件（DSH 的 `llm/retry`、
  `llm/retry-started`）；
- **工具不重试**：工具可以声明 `idempotent=True`，但当前 loop 不会重复执行工具，
  因此不会有重复副作用。

取消发生时，未派发的 tool call 会补一条 synthetic 结果
`Error: tool call aborted before dispatch`，保证消息序列里每个 call 都有对应结果。

## 模块

```text
app/
  contracts.py                    消息/工具/provider/事件/状态契约
  errors.py                       ProviderError + 错误码分类
  retry.py                        RetryPolicy + 退避计算（只重试模型请求）
  inbox.py                        Inbox：steering / follow-up 队列
  compaction.py                   CompactionPolicy + 摘要请求 + 区域切分
  memory.py                       长期记忆：有界事实库 / 用户画像
  memory_tools.py                 Agent 自管理的记忆工具（add / replace / remove）
  session_search.py               历史会话全文索引（SQLite FTS5 + LIKE 兜底）
  trajectory.py                   Trajectory 记录与汇总（Phase G 评测）
  rag_tools.py                    RAG 工具：search_spec / read_context + 引用渲染
  cli.py                          最小 CLI：一个问题 → 一次 run
  loop.py                         run_agent：turn 外层 + step 内层
  session.py                      JSONL 事件日志（turn/start、step/start、message、step/end、turn/end）
  tools.py                        ToolRegistry + FunctionTool
  prompts.py                      ReAct 系统提示词（只影响模型行为，不承担停止）
  providers/
    mock.py                       脚本化 provider，测试用
    openai_compatible.py          OpenAI-compatible chat completions（transport 可注入）
tests/
```

会话存的是事件而不是裸消息列表，重新加载时既能重建 `messages`，也能看出
turn/step 边界以及最后一个 turn 是否中断（`SessionTimeline.last_turn_open`）。

## 用法

```python
import asyncio

from app.contracts import AssistantTurn, Message, ToolResult
from app.loop import RunOptions, run_agent, user_message
from app.providers import MockProvider
from app.session import JsonlSessionStore
from app.tools import FunctionTool, ToolRegistry

registry = ToolRegistry().register(
    FunctionTool(name="echo", description="Echo", handler=lambda args, ctx: ToolResult(content=str(args["text"])))
)
provider = MockProvider(
    [AssistantTurn(message=Message(role="assistant", content="done"), stop_reason="stop")]
)

result = asyncio.run(
    run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            messages=[user_message("go")],
            session=JsonlSessionStore("session.jsonl"),
            cancel_event=asyncio.Event(),          # 可选
            on_event=lambda event: print(event),   # 可选
        )
    )
)
print(result.status, result.turns, result.steps)
```

## 开发

```powershell
cd apps/agent-runtime
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,http]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

Python >= 3.12；默认测试使用脚本化 provider 和本地 HTTP/SSE 服务，不调用付费模型。
固件行为测试需要可用的主机 C 编译器，可通过 `CC` 指定。真实模型验收单独手动执行，
不包含在默认 pytest 中。

## 边界 / 与 DSH 的有意简化

这一版**不是** DSH 的完整复刻，差异如下：

| DSH | 本包 |
|---|---|
| ~~turn/step 事件日志~~ | 已实现：外层 turn / 内层 step 双层循环，事件逐条落盘；但 turn 序号每次 run 从 1 起、step 跨 turn 累计（不接续原会话序号） |
| ~~并行工具 + exclusive barrier~~ | 已实现：有界并发池 + exclusive 屏障，默认 exclusive |
| ~~steering / follow-up~~ | 已实现（Inbox：next-step / next-turn）；队列是进程内的，未认领的输入不随会话恢复 |
| ~~session tree / compaction~~ | 已实现（树形节点 + 阈值压缩 + 摘要，branch/fork 未做） |
| `compaction-basic` 的重压缩检查 / context-overflow 恢复 | 落地后重新计量与有限重试已实现（默认最多两次，仍超阈值则警告并继续）；真实 context-overflow 恢复未移植 |
| `compaction/start` → `end` 事务、generation 校验、源事件 seq 范围 | **未移植**：本地用 `CompactionEntry` + `kept_messages` 条数重放，不是按源事件 seq 记录替换范围 |
| ~~`LlmAdapter.stream()` 单入口 + StreamChunk 协议~~ | 已对齐：唯一入口 + block 生命周期 + `usage` / `finish` chunk |
| `ContentBlock` 六类（text / reasoning / image / file / tool-call / tool-result） | **只有 text / reasoning / tool-call**：本地消息模型没有多模态与结构化 tool-result block，工具结果走 `role="tool"` 消息 |
| `finish.replayState` / `ReplayEnvelope` | **未实现**：本地 session 重放走 messages，不需要 adapter 私有的无损响应状态 |
| `tools/schema.ts` 的完整 schema 校验 | **有界子集**：`app/tools.py::VALIDATION_KEYWORDS` 列出的关键字在派发前真正校验，`ANNOTATION_KEYWORDS`（title/description/default/examples…）按上游当纯注解；子集外的关键字（`oneOf` / `$ref` / 元组式 `items`）**拒绝派发**并给出原因，不会「透传给模型就算校验过」 |
| 原始 chunk 持久化（`assistant/attempt.stream`） | 未持久化原始 chunk；只落可见前缀与 attempt 摘要 |
| `agent/turn-stopping` 扩展点 | 未实现；需要时用 `cancel_event` 取消 |
| 多 provider 路由 / 配置目录 | 后置（retry 已实现，但只针对模型请求） |

其他边界：不含 UI、MCP、OS 沙箱。子 Agent 范围见上文，不能称整个 DSH 等价；
任务与续接组件保留上游原文提示词，来源和许可见上述说明。
