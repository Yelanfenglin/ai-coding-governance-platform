# 运行观测与故障治理

[首页](../README.md) · [原理](architecture.md) · [运行](development.md)

**从指标告警定位异常，用日志关联任务，再沿 Trace 找到具体调用。**

## 观测覆盖

| 层次 | 记录与定位 |
|---|---|
| 任务 | 总耗时、在途任务、最近进展事件，以及任务结果和交付检查状态 |
| 模型与工具 | 实际请求与工具派发、错误码、耗时、首事件/首字延迟和重试终局 |
| 输入与输出 | 缓存/未缓存输入、输出与推理用量，区分对话和摘要调用 |
| RAG | 检索耗时、命中数量、重排降级、入库队列与 PostgreSQL/Milvus 就绪状态 |
| 基础服务 | 主机、容器和数据库指标，服务连通与就绪探测 |

Prometheus 采集指标，Grafana 展示任务、模型、检索与资源看板；Alloy 将结构化日志送入 Loki。日志中的 `trace_id` 关联 Phoenix 调用链，任务结果与交付记录保留同一追踪标识。

## 从排障到优化

- **慢或卡住**：先看任务时长与最近事件，再拆分模型等待、首字延迟、工具执行和检索耗时。
- **连续失败**：沿错误码和 Trace 关联具体请求，观察重试次数、熔断拒绝与半开探测，定位模型或 RAG 依赖。
- **输入成本上升**：检查实际请求次数、重试与缓存/未缓存输入，定位重复检索和过长工具输出，再用固定任务复测策略。

已有优化包括符号级源码定位、符号结果与诊断输出压缩、按需证据补读和上下文压缩；量化对照见[首页评测](../README.md#效果与验证)。

## 故障处理

模型请求对空响应、超时和未收尾的 SSE 流执行错误分类与有限重试，采用指数退避并遵守 `Retry-After`。模型与 RAG 依赖分别维护熔断状态，冷却后由下一请求发起单次半开探测。

告警覆盖服务不可达、依赖未就绪、任务与模型长时间等待、检索延迟和熔断拒绝，随规则携带定位提示。运行中的取消会回收已启动工具并保留已完成结果；交付时由宿主复验最终文件。

当前为本地单 worker 部署，告警默认只在站内显示。外部通知渠道由部署者配置。

<details>
<summary>服务入口与启动</summary>

| 本地入口 | 内容 |
|---|---|
| `http://127.0.0.1:3000` | Grafana 服务、资源、任务、模型、RAG 与站内告警 |
| `http://127.0.0.1:9090` | Prometheus targets 与指标查询 |
| `http://127.0.0.1:6006` | Phoenix 任务、模型、工具与检索调用链 |

Agent 安装 `.[web,observability]`，RAG 安装 `.[observability]`。从仓库根目录执行：

```powershell
$env:PYTHONNOUSERSITE='1'
New-Item -ItemType Directory -Force output/service_logs, output/monitoring_secrets
& apps/agent-runtime/.venv/Scripts/python.exe -c "from pathlib import Path; import secrets; p=Path('output/monitoring_secrets/grafana_password'); p.write_text(secrets.token_urlsafe(32)) if not p.exists() else None"
docker compose -f infra/observability/compose.yaml up -d
docker compose -f infra/monitoring/compose.yaml up -d
```

在两个服务各自的 `.env` 中设置日志路径和 Trace 端点后重启：

```dotenv
# Agent
MONITORING_LOG_PATH=../../output/service_logs/agent.jsonl
# RAG
MONITORING_LOG_PATH=../../output/service_logs/rag.jsonl
# 两个服务
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:6006/v1/traces
```

Grafana 管理密码保存在本机文件中，默认允许回环地址上的匿名只读访问；告警渠道在配置后解除静默。指标与日志保留策略见 [监控 Compose](../infra/monitoring/compose.yaml)。

配置 `PHOENIX_PROJECT_ID` 后，Loki 日志中的 Trace 标识可跳转到对应 Phoenix 项目。

</details>

<details>
<summary>主机、数据库和健康检查</summary>

Windows 与 PostgreSQL 分别使用 `windows_exporter` 和 `postgres_exporter`；容器使用 cAdvisor，Milvus 提供自身指标。

数据库监控账号可通过 [bootstrap_postgres.py](../infra/monitoring/bootstrap_postgres.py) 创建：交互输入管理员密码，独立监控凭据保存到忽略目录，授予 `pg_monitor` 与默认只读事务，不修改业务表。

Agent 健康接口为 `/api/agent/health`，RAG 为 `/api/rag/health`。RAG 的 HTTP 200 仍可能 `ready=false`，探测检查 JSON 就绪状态；配置 API key 时携带 `X-API-Key`。

容器访问宿主使用 `host.docker.internal`；Agent 探测保留 `Host: 127.0.0.1:8081`。健康检查和看板刷新不主动调用模型。

[Milvus Compose](../apps/rag-api/infra/milvus/compose.yaml) 管理 `rag-milvus-data` 数据卷，宿主端口为 19531。备份或恢复前停止 Milvus/etcd，保留完整数据卷；服务就绪以 RAG 健康结果为准。

</details>

<details>
<summary>计量与告警口径</summary>

任务结果与交付检查分别记录；模型请求包含失败尝试，工具按实际派发计量。输入用量包含缓存部分，缓存与未缓存用于分析输入构成，缺失用量保持未知。

重试恢复率按 `recovered/(recovered+failed)` 计算，无终局样本显示无数据。取消与恢复分别计量；熔断拒绝发生在网络请求发出前。

熔断告警要求状态打开、近两分钟有拒绝增量并持续一分钟。指标不携带任务 ID、文件路径、错误正文或 chunk ID，定位由日志和 Trace 承担。模型输入 tokens 不与任务摘要重复相加，原始 usage 不等于账单。

默认不导出问题正文、推理正文、记忆、工具参数/结果和鉴权头。SDK 导出失败不触发任务重试，本地交付记录独立保存。整机故障由机器外探测负责。

配置与采集公式见 [infra/monitoring](../infra/monitoring/)；观测 Compose 另保留可选的 Uptime Kuma。

</details>

<details>
<summary>实现与验证入口</summary>

| 能力 | 实现 | 验证 |
|---|---|---|
| 调用追踪与日志关联 | [observability.py](../apps/agent-runtime/app/observability.py)、[数据源配置](../infra/monitoring/grafana/provisioning/datasources/default.yaml) | [追踪与用量关联](../apps/agent-runtime/tests/test_observability.py) |
| 请求、工具与任务计量 | [Agent 指标](../apps/agent-runtime/app/metrics.py)、[RAG 指标](../apps/rag-api/app/observability/metrics.py) | [实际派发与重试计量](../apps/agent-runtime/tests/test_metrics.py) |
| 重试与依赖隔离 | [retry.py](../apps/agent-runtime/app/retry.py)、[circuit.py](../apps/agent-runtime/app/circuit.py) | [入口恢复](../apps/agent-runtime/tests/test_retry_entrypoints.py)、[熔断状态机](../apps/agent-runtime/tests/test_circuit.py) |
| 输入与上下文优化 | [符号检索](../apps/agent-runtime/app/source_symbols.py)、[上下文压缩](../apps/agent-runtime/app/compaction.py) | [上下文压力与计量](../apps/agent-runtime/tests/test_context_pressure.py) |
| 服务与业务告警 | [采集配置](../infra/monitoring/prometheus.yaml)、[告警规则](../infra/monitoring/grafana/provisioning/alerting/rules.yaml) | 按规则查看服务、任务、检索与熔断状态 |

</details>
