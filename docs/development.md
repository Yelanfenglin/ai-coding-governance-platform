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

`evaluation.runner` 复用冻结 case 与独立行为检查，不调用模型：

```powershell
.venv/Scripts/python.exe -m evaluation.runner check --case evaluation/cases/multi_module_change.json --workspace <工程目录> --out <检查目录>
.venv/Scripts/python.exe -m evaluation.runner score --case evaluation/cases/multi_module_change.json --delivery <任务的delivery.json> --workspace <对应工程> --out <评分目录>
.venv/Scripts/python.exe -m evaluation.runner batch --runs <runs.jsonl> --out <批次目录>
```

Web 任务的交付文件位于工程 `.agent-runtime/web/tasks/<task-id>/delivery/`。同一工程持续修改时，每轮另存 checkpoint；历史评分必须使用该轮最终版本。

批次清单每行声明 `case`、`delivery`、`workspace`，相对路径以清单所在目录为基准。确认无人工代码修复时才设置 `human_interventions=0`，未知保持空值。评分核对冻结起点、最终版本、任务状态、清单、保护文件与独立行为；缺失运行与失败保留在分母。

RAG 的冻结题库、必要证据组和评分入口见 [RAG 配置](../apps/rag-api/README.md)。模型请求、运行日志、原始验收报告及简历材料均保存在仓库外或本地忽略目录。
