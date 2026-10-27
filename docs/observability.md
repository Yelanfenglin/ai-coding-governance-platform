# 监控与运维

[首页](../README.md) · [原理](architecture.md) · [运行](development.md)

**Grafana 看状态，日志查事件，Phoenix 沿 Trace 找到具体调用。**

| 本地入口 | 内容 |
|---|---|
| `http://127.0.0.1:3000` | 服务、资源、任务、模型、RAG 与站内告警 |
| `http://127.0.0.1:9090` | Prometheus targets 与指标查询 |
| `http://127.0.0.1:6006` | Phoenix 任务、模型、工具与检索调用链 |

## 怎样读这些数据

- **任务与检查分开**：运行结束后，还要看交付检查是否通过。
- **用量按实际请求记录**：输入已包含缓存部分；未知用量保持未知。
- **重试与拒绝分开**：被熔断拒绝的调用没有发出网络请求，恢复需下一请求探测。

当前为本地单 worker 部署，告警默认只在站内显示。外部通知渠道由部署者配置。

<details>
<summary>启动监控</summary>

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
<summary>计量、告警与数据边界</summary>

模型每次实际请求包含失败尝试，工具只记录已派发执行。取消不算恢复；重试恢复率按 `recovered/(recovered+failed)` 计算，无终局样本显示无数据。

熔断告警要求状态打开、近两分钟有拒绝增量并持续一分钟。指标不携带任务 ID、文件路径、错误正文或 chunk ID，定位由日志和 Trace 承担。模型输入 tokens 不与任务摘要重复相加，原始 usage 不等于账单。

默认不导出问题正文、推理正文、记忆、工具参数/结果和鉴权头。SDK 导出失败不触发任务重试，本地交付记录独立保存。整机故障由机器外探测负责。

配置与采集公式见 [infra/monitoring](../infra/monitoring/)；观测 Compose 另保留可选的 Uptime Kuma。

</details>
