# 开发与演示

## 本地检查

从仓库根目录运行：

```powershell
npm test
npm run build
npm run test:e2e
```

Python 命令使用各模块自己的虚拟环境，并设置 `PYTHONNOUSERSITE=1`。Agent 测试覆盖工具调度、取消、会话、记忆、压缩、HTTP/SSE 与交付。浏览器测试使用独立工程和脚本 HTTP 对端，不调用付费模型。

RAG 在 `apps/rag-api` 下运行：

```powershell
$env:PYTHONNOUSERSITE='1'
.venv/Scripts/python.exe -m pytest -q -m "not integration"
```

真实数据库与向量库测试按 integration 标记单独运行；真实模型任务与自动化接线测试分开记录。

## C 源码定位

工作区默认提供两个只读工具，Web 与 CLI 共用；启用子 Agent 时也开放给只读研究工具集。

- `find_symbols`：`query` 默认先按完整名称精确匹配，没有精确结果时再包含匹配；`match=exact/contains` 可显式选择。可用 `kind=function/macro/type` 筛选，返回定义或声明、签名、文件和行号；`include_body=true` 附带最多 40 行原文。
- `find_symbol_references`：`query` 精确匹配标识符，返回调用或引用、所属函数及附近代码。定义和声明也会作为相应类型的匹配返回。

例如先以 `find_symbols(query="Init", kind="function", path="src")` 找入口，再按完整函数名
调用 `find_symbol_references` 找调用方；需要更多实现细节时，用已有 `read_file` 读取返回的范围。
默认返回最多 20 条，`max_results` 可设为 1～100；`total_matches` 与 `truncated` 表示截断情况，
命中过多时缩小查询或路径。

完整名称查询会保留全部同名结果，并通过 `omitted_partial_matches` 告知过滤了多少包含匹配，
需要查包装函数或相关名称时可用 `match=contains`。默认诊断仅返回命中文件的摘要；
`files_with_parse_issues` 与 `unreadable_files` 保留全扫描的问题计数，避免将无结果误认成
完整解析成功。需要具体错误位置时，指定文件并设置 `include_diagnostics=true`；
该参数可展开扫描诊断，最多返回 20 个文件、每文件最多 5 处解析位置。

解析使用 [py-tree-sitter](https://tree-sitter.github.io/py-tree-sitter/) 和官方
[C grammar](https://github.com/tree-sitter/tree-sitter-c)。支持 `.c/.h`（含大写扩展名），
按每次调用时的实际文件读取，编辑、增加或删除文件后再次查询即可。默认跳过 `.git`、
`.venv`、`node_modules`、`__pycache__`、`.agent-runtime`、`.agent-demo-backups`、`output`、
`build`、`dist` 目录；显式指定其中的 C 文件仍可读取，路径须在工作区内。

这是语法定位：注释和字符串不作为引用，条件编译分支全部保留，不展开宏或解析函数指针目标，
跨文件同名符号需结合文件和调用方核对。厂商扩展或宏修饰符可能使部分声明无法识别，
结果的 `diagnostics` 会报告解析问题。它不替代编译器的语义分析和工程验收。

## 固定工程演示

示例任务：在 M9P6362 V1.16 的 PWM0＋USART0＋ADC＋TC1 工程上，把 PWM0 从 30% 改成 40%，保持 20kHz；USART0 从 115200 改成 57600，保留非阻塞收发、接收中断与原有业务。

先在仓库根目录 `npm run build`，按 README 配好 Agent 与 RAG。主机行为检查需要 GCC，非 PATH 安装可用 `CC` 指定，编译器所需 DLL 目录也应加入 PATH。

在 `apps/agent-runtime` 下用 PowerShell 7 准备独立工程：

```powershell
$env:PYTHONNOUSERSITE='1'
$demoRun = Join-Path (Get-Location).Path ("output/demo-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$demoProject = Join-Path $demoRun 'project'
New-Item -ItemType Directory -Path $demoProject | Out-Null
Copy-Item -Path 'evaluation/fixtures/pwm_usart_adc/*' -Destination $demoProject -Recurse
$casePath = (Resolve-Path 'evaluation/cases/multi_module_change.json').Path
$verificationPath = Join-Path $demoRun 'verification.json'
@{
  command = @('{python}', '-m', 'evaluation.runner', 'check', '--case', $casePath,
              '--workspace', '{workspace}', '--out', '{workspace}/.agent-runtime/checks')
  cwd = '{workspace}'
  timeout = 90
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $verificationPath
.venv/Scripts/python.exe -m app.web --env-file .env --workspace $demoProject --verification $verificationPath --product M9P6362 --version V1.16
```

启动前停止占用 8081 的旧 Agent 服务。打开 `http://127.0.0.1:8081`，将 `evaluation/cases/multi_module_change.json` 的 `question` 作为任务输入。该任务会真实调用模型与检索服务。

演示关注四处：

1. Agent 读取现有工程并按需查询规格，实际运行计算。
2. 修改后调用 `verify_project`，根据失败输出修复，保留原有受保护文件。
3. 完成后查看“工程审阅”：实现理由、源码差异、规格页码与检查版本相互关联。
4. 阅读实际答复后再决定追问，例如核对某项计算前提或纠正需求，不预写后续多轮答案。

对实现说明的审阅应区分规格明示与计算推导，并按实际工具输出描述检查范围。主机编译或模拟已执行时，不能概括成“仅静态核对”。

## 交付评分与离线重评

芯片迁移用例在 `evaluation/cases/chip_migration_adc.json`。复制
`evaluation/fixtures/chip_migration_adc` 为独立工作区，保留 `source/` 与 `target/`
结构；沿用上面的宿主验收配置，将 `casePath` 指向这个用例，再以普通
`--workspace` 模式运行用例的 `question`。工作区提供迁移 Skill 的摘要与读取工具，
由模型按需加载，使用现有文件、规格检索和命令工具完成适配。

该用例将 M9P6362 的单次非阻塞 ADC 模块迁入 M9F6820 TC1 工程。同一份 C 行为检查
分别编译并执行源与目标模块；目标另外验证工程登记、系统配置及原有 TC1 中断和心跳。
这是主机侧模块验收，厂商编译和板卡运行分别记录。

### ADC 操作序列探索（可选）

在 `apps/agent-runtime` 安装 `.[evaluation]` 后，可给原迁移检查增加序列探索：

```powershell
$env:PYTHONNOUSERSITE='1'
.venv/Scripts/python.exe -m pip install -e ".[evaluation]"
.venv/Scripts/python.exe evaluation/checks/adc_migration.py --workspace <迁移工程目录> --out <工程外检查目录> --explore-sequences --sequence-examples 300 --sequence-steps 30 --sequence-seed 20261025
```

检查器编译源、目标 C 模块，从空闲、转换中、完成但未轮询、结果就绪四个正常状态
开始组合 `Start`、`Poll`、`Read`、空指针读取和转换完成事件，逐步比较返回值、输出参数
与 ADS。转换完成仅在 ADS 置位时注入；随机值覆盖 16 位结果寄存器。
[Hypothesis](https://hypothesis.readthedocs.io/en/latest/stateful.html) 负责生成与缩减，
状态到达过程也保留在反例中。`max_examples` 限制搜索样本量，失败后的缩减会增加执行次数。

原有固定检查继续验证型号配置、工程完整性和 TC1 业务。序列检查作为补充，默认关闭；
无差异表示在本次搜索预算内未发现差异。源实现是比较基准，双方共有的问题不能靠差分识别。
主机寄存器替身不模拟 ADC 时序、模拟精度或厂商 ABI，也不检查私有状态变量。

失败时，命令标准输出给出缩减序列、首次差异和重放命令，完整轨迹保存在
`sequences/counterexample.json`；可直接放入已有 `--verification` 的宿主固定命令，
由 `verify_project` 将实际输出返回模型。构建与报告目录须在被检查工程外，避免改变工程版本。
修复后先重放反例，再运行完整检查：

```powershell
.venv/Scripts/python.exe evaluation/checks/adc_sequences.py --workspace <迁移工程目录> --out <工程外重放目录> --replay <反例文件.json>
```

反例为普通 JSON，重放不重新搜索，也不依赖随机种子；完整搜索的复现需要相同 Hypothesis
版本与参数。`sequence_checks.json` 分别记录搜索/缩减次数与反例确认/重放次数，以及版本、
搜索参数和耗时；缩减结果不承诺全局最短。

`evaluation.runner` 复用冻结 case 与独立行为检查，不调用模型：

```powershell
.venv/Scripts/python.exe -m evaluation.runner check --case evaluation/cases/multi_module_change.json --workspace <工程目录> --out <检查目录>
.venv/Scripts/python.exe -m evaluation.runner score --case evaluation/cases/multi_module_change.json --delivery <任务的delivery.json> --workspace <对应工程> --out <评分目录>
.venv/Scripts/python.exe -m evaluation.runner batch --runs <runs.jsonl> --out <批次目录>
```

Web 任务的交付文件位于工程 `.agent-runtime/web/tasks/<task-id>/delivery/`。同一工程持续修改时，每轮另存 checkpoint；历史评分必须使用该轮最终版本。

### 交付后复验

工作台的“工程审阅 → 验收”提供“重新验收当前交付”。它仅执行宿主配置的检查，
不调用模型；复验期间同一服务不启动其他工程任务。每次结果单独保存到交付目录的
`rechecks/`，原任务状态、退出码、`delivery.json` 与事件记录保留。

复验前核对当前工程版本与原交付一致，且宿主命令、工作目录、超时参数与原最终检查相同。
工程或配置已改变时拒绝复验；运行过程中改变工程也不能判为通过。旧复验在工程变化后
重新加载时标为过期。可用 CLI 对已有交付做相同操作：

```powershell
$env:PYTHONNOUSERSITE='1'
.venv/Scripts/python.exe -m app.reverification --workspace <工程目录> --delivery <delivery.json> --verification <宿主验收配置.json>
```

“原交付失败、后来复验通过”作为两条结果展示，不自动重试、不修改原成绩，也不将复验
通过改写为该轮模型自主交付成功。

批次清单每行声明 `case`、`delivery`、`workspace`，相对路径以清单所在目录为基准。确认无人工代码修复时才设置 `human_interventions=0`，未知保持空值。评分核对冻结起点、最终版本、任务状态、清单、保护文件与独立行为；缺失运行与失败保留在分母。

RAG 的冻结题库、必要证据组和评分入口见 [RAG 配置](../apps/rag-api/README.md)。模型请求、运行日志、原始验收报告及简历材料均保存在仓库外或本地忽略目录。
