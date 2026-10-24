# 系统运行监控

Grafana 是本地运维入口，Prometheus 采集时序指标，Loki/Alloy 收集结构化运行日志。
Phoenix 保留任务调用链，使用 trace ID 从日志定位请求。会话和交付记录仍在原位置。

| 入口 | 用途 |
|---|---|
| `http://127.0.0.1:3000` | 系统总览、服务与资源、任务与模型、RAG 与入库，以及告警 |
| `http://127.0.0.1:9090` | Prometheus 采集状态与指标查询 |
| `http://127.0.0.1:6006` | Phoenix 任务调用链 |

## 部署系统监控

Agent 安装 `pip install -e ".[web,observability]"`，RAG 安装 `pip install -e ".[observability]"`。
使用各模块自己的虚拟环境，并设置 `PYTHONNOUSERSITE=1`。

在仓库根目录创建本地日志目录及管理员密码文件；密码文件不入库：

```powershell
New-Item -ItemType Directory -Force output/service_logs, output/monitoring_secrets
& apps/agent-runtime/.venv/Scripts/python.exe -c "from pathlib import Path; import secrets; p=Path('output/monitoring_secrets/grafana_password'); p.write_text(secrets.token_urlsafe(32)) if not p.exists() else None"
docker compose -f infra/observability/compose.yaml up -d
docker compose -f apps/rag-api/infra/milvus/compose.yaml up -d
docker compose -f infra/monitoring/compose.yaml up -d
```

Grafana 允许回环地址上的匿名只读访问，管理员密码保存在本机文件中。容器有独立数据卷，
指标保存到 15 天或 2GB 上限先到为止；Loki 目标保留期为 15 天。
告警规则、看板与数据源使用文件 provisioning。当前通知策略全天静默，只在站内显示状态，
不发送邮件或第三方消息。需要通知时由部署方配置真实渠道并解除相应静默。

在 Agent、RAG 的 `.env` 分别设置，再重启进程：

```dotenv
# Agent
MONITORING_LOG_PATH=../../output/service_logs/agent.jsonl
# RAG
MONITORING_LOG_PATH=../../output/service_logs/rag.jsonl
```

路径相对各自服务的工作目录。日志按文件大小轮转；Alloy 只读挂载日志目录。
日志不记录问题正文、模型思考、工具输入输出或鉴权头。

Windows 上 RAG 的 Psycopg 异步连接需要 SelectorEventLoop。在 `apps/rag-api/` 下启动：

```powershell
$env:PYTHONNOUSERSITE='1'
.venv/Scripts/python.exe -c "import asyncio,uvicorn; c=uvicorn.Config('app.main:app',host='127.0.0.1',port=8000,env_file='.env',loop='none'); asyncio.run(uvicorn.Server(c).serve(),loop_factory=asyncio.SelectorEventLoop)"
```

这一设置仅用于 RAG 进程；Agent 继续使用支持本地异步子进程的默认事件循环。

### 主机和数据库

主机使用官方 [windows_exporter](https://github.com/prometheus-community/windows_exporter)，
可运行已验证版本的独立可执行文件，不要求修改应用去采集 Windows 性能计数器：

```powershell
& <windows_exporter.exe> --web.listen-address=127.0.0.1:9182 `
  --collectors.enabled=cpu,logical_disk,memory,net,os,service,system,process `
  --collector.service.include='postgresql.*|com.docker.service' `
  --collector.process.include='(python|postgres|node|com.docker.backend|windows_exporter).*' `
  --no-collector.process.cmdline
```

Windows 主机资源与 cAdvisor 的 Docker Linux 容器资源分开显示。cAdvisor 使用官方镜像与
所需的只读挂载；Docker socket 与特权权限仅供本地容器采集，未发布管理端口。

PostgreSQL 的 TCP 连通性和 RAG 的 `SELECT 1` 检查无需新增数据库权限。
连接数、数据库大小等深度指标使用官方
[postgres_exporter](https://github.com/prometheus-community/postgres_exporter)。
部署前由数据库管理员执行 `infra/monitoring/postgres-monitor.sql`，设置专用账号密码；
脚本只创建监控角色、授予 `pg_monitor` 并设置只读事务，不改业务表。
也可在仓库根目录运行下面的引导脚本，仅在本机输入已有管理员密码；它会生成并保存
独立的监控密码及连接元数据到被忽略的 `output/monitoring_secrets/`：

```powershell
$env:PYTHONNOUSERSITE='1'
& apps/rag-api/.venv/Scripts/python.exe infra/monitoring/bootstrap_postgres.py
```

该脚本调用 PostgreSQL 自带的 `psql` 读取管理员密码，不接收或保存管理员密码。
若客户端不在 PATH 中，通过 `--psql <psql.exe 路径>` 指定。已有监控凭据时不重置密码。

```powershell
psql -h 127.0.0.1 -U <数据库管理员> -d rag_kb -f infra/monitoring/postgres-monitor.sql
$env:DATA_SOURCE_URI='127.0.0.1:5432/rag_kb?sslmode=disable'
$env:DATA_SOURCE_USER='embedded_monitor'
$env:DATA_SOURCE_PASS_FILE='<本机密码文件>'
& <postgres_exporter.exe> --web.listen-address=127.0.0.1:9187
```

采集配置排除其他业务数据库的指标。“PG 深度采集”同时检查 HTTP 可达性、`pg_up` 和
SQL 抓取错误，避免 HTTP 在线但采集失败时仍显示正常。看板展示数据库大小、当前连接、
事务速率、shared buffers 命中比、锁数量和死锁；无活动时的速率分母为零，不伪造命中比。
其余数据库健康检查仍独立工作，不把缺少 exporter 当成数据库宕机。
Milvus 使用自身 `9091/metrics`，Prometheus 接入已有 RAG Docker 网络，不修改向量集合。

## 采集与告警口径

- Agent `/metrics`：任务状态与宿主检查状态分别计数；模型每次实际请求含失败尝试；
  工具只计已派发的执行，取消不会计为成功。任务、模型和工具有各自的活跃数与时长。
- RAG `/metrics`：HTTP 与检索阶段请求、错误、延迟；`/search` 命中数量与降级；
  入库等待、活跃 worker、执行耗时及失败阶段。零命中是正常结果，单独统计。
- HTTP 指标由 `prometheus-fastapi-instrumentator` 采集，路由使用模板，健康检查和指标请求
  不进入业务流量统计。Agent 仅为无 Origin 的 Docker 主机 GET `/metrics` 放行监控访问，
  任务与文件 API 的本机同源校验不变。
- 指标标签不包含任务 ID、会话 ID、文件路径、错误正文或 chunk ID。日志与 Trace 负责定位。
- 指标属于当前服务进程，重启后的计数重置由 Prometheus `rate/increase` 处理。
  当前按单 worker 部署；独立 CLI 任务仍有日志和 Trace，退出后的内存指标不补写到 Web 进程。

默认告警覆盖入口故障、RAG 未就绪、依赖连通性、采集停止、磁盘与内存压力、模型错误与
首响应延迟、任务运行错误、交付检查失败、检索变慢，以及入库 worker 停止。
“任务缺少进展事件”是需要检查的状态，不证明死锁，也不会自动终止任务。
低流量的失败率与延迟告警要求足够样本；阈值在 provisioning 文件中配置，不作为生产 SLO。

“任务与模型”看板另展示依赖熔断状态、拒绝数和重试恢复率。后两项为当前进程累计值，
进程重启后重新累计。恢复率按进入过重试的逻辑请求计算 `recovered / (recovered + failed)`，
取消不进分母；没有已结束样本时显示无数据。拒绝数只统计未发出网络请求的调用。
熔断告警要求状态为打开、最近两分钟有拒绝增量，并持续一分钟；空闲时仅有打开状态不会
单独触发该告警。冷却结束后仍需下一次请求探测，恢复耗时不等于依赖实际停机时长。

部署后检查 Prometheus targets，运行一项任务核对请求与 token，再在空闲时验证
服务停止、依赖中断的触发和恢复。Grafana 告警状态变化写入 Loki。
健康探测、抓取指标、刷新看板不会主动调用模型或 embedding。

这套配置监控当前单机部署；整机断电时，需要部署在机器外的探测与通知才能独立告警。
Uptime Kuma 保留原数据，不作为这套运维入口的前置条件。

## 调用链配置

运行链路使用 OpenTelemetry，遵循 OpenInference 属性约定，默认没有远程导出。
本地 Phoenix 接收和展示 Trace。

在 Agent、RAG 各自的虚拟环境安装 `pip install -e ".[observability]"`。
从仓库根目录启动观测服务：

```powershell
docker compose -f infra/observability/compose.yaml up -d
```

在 Agent 和 RAG 的 `.env` 中设置，然后重启各自服务：

```dotenv
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:6006/v1/traces
```

Agent 的 Web 入口通过 `--env-file .env` 加载配置。直接使用 CLI 时，先将该变量设置到
进程环境。`OTEL_SDK_DISABLED=true` 可关闭导出。可通过 SDK 标准的
`OTEL_EXPORTER_OTLP_TIMEOUT`、`OTEL_BSP_*` 环境变量调整传输和批处理。

Phoenix 在 `http://127.0.0.1:6006`，项目为 `embedded-assistant`；Uptime Kuma 在
`http://127.0.0.1:3001`，首次打开需创建本机管理账号。两个服务只发布回环端口，
数据保存在 Docker 命名卷中。通知渠道由使用方配置。

### Trace 与模型用量

每次 CLI/Web 任务有独立的 `agent.task`，Web 会话使用 `session.id` 关联多轮。
启用采集时，任务响应和交付记录中的 `trace_id` 可定位到对应执行。

| 记录 | 内容 |
|---|---|
| `agent.task` | 任务 ID、会话、运行状态、交付检查状态与退出码 |
| `agent.run` | 主/子 Agent 运行，保留工具委派产生的父子关系 |
| `llm.request` | 每次实际模型请求，包括失败重试；请求/响应模型、推理配置、结束原因 |
| `tool.*` | 调用 ID、工具名、实际执行区间和错误状态；并行调用分别记录 |
| `rag.client.*` → `rag.http` | W3C `traceparent` 贯通 Agent 与 RAG 服务 |
| `rag.search` 及其子阶段 | 文档范围解析、embedding、Milvus、标准 chunk 读取、BM25；命中数与证据 ID |
| `context.compaction` | 压缩调用区间；摘要模型请求带 `purpose=compaction` |
| `project.verify` | 宿主检查区间、阶段及通过状态，位于任务结束之前 |

运行事件补充步骤、重试、记忆召回数量和压缩前后 token 数。Web 事件增加观察时间；
任务记录保存开始与结束时间。

模型输入 tokens 包含缓存命中部分，缓存命中与未命中分别记录，不能再加到输入总数。
服务端未返回缓存或推理用量时保持未知；不把未知记为零。首个流事件与首段回答文本
分别计时，推理模型的思考输出不算首段回答。原始 usage 不等于账户账单；目前不计算金额。
任务级摘要不重复登记 LLM token 指标，避免与请求级统计相加后重复计费。

`completed` 表示运行结束；交付检查单独记录 `passed/failed/unverified`。
空检索结果、预算耗尽、用户取消和服务故障分别保留其状态。
采集不导出原始问题、思考正文、记忆内容、工具参数/结果或 HTTP 鉴权头。
批处理和发送由 SDK 负责，导出失败不触发任务重试。

### 健康检查

- Agent：`GET /api/agent/health`。检查本地服务可响应，返回当前任务、最近事件与时间；
  不调用模型，也不宣称外部模型可用。
- RAG：`GET /api/rag/health`。检查 PostgreSQL/Milvus，模型部分为已有配置/状态检查，
  不消耗推理额度。配置 API key 时需传 `X-API-Key`。

Kuma 使用 HTTP JSON Query，表达式为 `ready`，期待值为 `true`。RAG 可能返回
HTTP 200 且 `ready=false`，只判断 HTTP 状态会漏掉依赖故障。
容器访问宿主服务时使用 `host.docker.internal`；Agent 请求头 `Host` 设置为
`127.0.0.1:8081`，保持本机 Host 校验。
具体端口、RAG 凭据按部署配置，不写入共享的 Compose 文件。
监控栈的 blackbox 同样检查 JSON `ready`，而不是只检查 HTTP 200。RAG 启用 API key 时，
需在本机私有探测配置中传入 key，不能写进共享配置。Phoenix 日志直达链接所需的项目 ID
通过监控 Compose 的本机 `.env` 中 `PHOENIX_PROJECT_ID` 配置。

## 实现参照

- DSH `aa8262e`：`packages/session/session-telemetry/src/coordinator.ts` 的采集与导出边界。
  本地使用持续 Span，不复制上游仅在反馈后上传日志的策略。
- [OpenHands `dd0892c`](https://github.com/OpenHands/software-agent-sdk/blob/dd0892c80fffb1e59bac77242adafbd8b1168649/openhands-sdk/openhands/sdk/llm/utils/telemetry.py)：
  请求生命周期与用量登记；`observability/laminar.py::RootSpan` 的显式上下文挂载。
  本地直接使用 OTel API，流式生成器在每次读取时挂载，跨 `yield` 不保留 context token。
- [Codex `de06fd0`](https://github.com/openai/codex/blob/de06fd0be4f465bc28c3111bdadaabda5dfdebcd/codex-rs/otel/src/events/session_telemetry.rs)：
  请求/工具耗时、缓存用量与实际推理配置的记录口径。

SDK 队列、导出器与展示后端均复用现成实现。Trace 是诊断记录，不代替本地交付记录，
也不承诺进程被强杀后队列中尚未发送的数据完整保存。
