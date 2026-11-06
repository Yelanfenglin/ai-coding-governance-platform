# 容器化部署与回滚演练记录

- 日期：2026-11-06
- 环境：Windows 11 · Docker 29.7.2 · Docker Compose v5.4.0
- 对象：`compose.yaml`（postgres / rag-api / agent-runtime；Milvus 复用外部 `rag-milvus` 实例）
- 脚本：`scripts/compose-release.ps1`（snapshot / rollback / status / release）

## 一、为什么要做这次演练

`compose.yaml` 里写了 healthcheck，也写了回滚用的镜像 tag 约定。但"写了"和"有效"是两件事：

- healthcheck 可能只探端口、不判就绪，那样它永远是 healthy，等于装饰；
- 回滚脚本可能从没被跑过，真出事时才发现参数写错。

所以这次把两件事都实际跑一遍，并记录结果。

## 二、演练一：健康检查能否发现依赖故障

### 步骤

1. `docker compose stop postgres`
2. 请求 `GET http://127.0.0.1:8080/api/rag/health`
3. 连续观察容器 `.State.Health.Status`

### 修复前的观测（说明问题真实存在）

依赖已经断开，接口如实返回：

```json
{"ready":false,"postgres":{"ok":false,"detail":"unavailable"},"milvus":{"ok":true,...}}
```

但容器在 120 秒内**始终显示 `healthy`**：

```
[10s] health=healthy failingStreak=0
[20s] health=healthy failingStreak=0
...
[120s] health=healthy failingStreak=0
```

**根因**：镜像里的 HEALTHCHECK 只执行 `urllib.request.urlopen(...)`，判据是"HTTP 请求是否成功"，而这个接口在依赖不可用时仍返回 HTTP 200，只是在响应体里标记 `ready:false`。健康检查没有解析响应体。

### 修复

两个 Dockerfile 的 HEALTHCHECK 改为解析响应并据此返回退出码：

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json,sys,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8080/api/rag/health',timeout=3)); sys.exit(0 if d.get('ready') else 1)"
```

### 修复后的观测

```
[0s]  health=starting  failingStreak=0
[10s] health=unhealthy failingStreak=0
```

容器由 `starting` 转 `unhealthy`；同时应用因启动期数据库校验失败而退出，`restart: unless-stopped` 使其进入重启。

恢复依赖后：

```powershell
docker compose start postgres
# [0s] health=healthy
```

容器自动回到 `healthy`，`/api/rag/health` 恢复 `ready:true`。

**结论**：健康检查现在能区分"进程活着"和"服务可用"，不再把依赖故障误判为健康。

## 三、演练二：按镜像 tag 回滚

### 前置

```powershell
.\scripts\compose-release.ps1 -Action snapshot -Version v1   # 修复前
.\scripts\compose-release.ps1 -Action snapshot -Version v2   # 修复后
```

自建镜像只有两个，postgres 使用上游固定 tag，不参与快照：

- `embedded-ai-coding-agent-runtime`
- `embedded-ai-coding-rag-api`

### 步骤与输出

```text
[1] 初始状态
agent-runtime   Up (healthy)
postgres        Up (healthy)
rag-api         Up (healthy)

[2] 模拟误发布：latest -> v1（旧版）
当前 rag-api 镜像: f50f911ba9e6   (v1 = f50f911ba9e6)

[3] 执行回滚
[rollback] embedded-ai-coding-agent-runtime:v2 -> :latest
[rollback] embedded-ai-coding-rag-api:v2 -> :latest
[rollback] 已按 v2 重建容器

[4] 回滚后
agent-runtime   Up (healthy)
postgres        Up (healthy)
rag-api         Up (healthy)
当前 rag-api 镜像: 006bd6ed861c   (v2 = 006bd6ed861c)

[5] 健康接口
rag-api ready=True
agent-runtime HTTP 200  X-Request-ID=8138258cd2d847aca2699b27f6e46cd1
```

回滚通过 `docker tag <image>:<version> <image>:latest` 加 `docker compose up -d --force-recreate --no-build` 完成；`--no-build` 保证回退不会触发重新构建。

### 脚本踩到的坑（已修）

首次执行时脚本在 recreate 阶段报 `exit=1`，但镜像 tag 其实已经切回、容器也已重建成功。原因是 PowerShell 在 `$ErrorActionPreference='Stop'` 下会把 docker 写到 stderr 的进度行当成终止性错误。修复方式：调用期间临时放宽 `$ErrorActionPreference`，只按退出码判成败。

## 四、结论与边界

**已验证**

- 健康检查能发现依赖故障并转为 `unhealthy`，依赖恢复后自动回到 `healthy`；
- 按镜像 tag 回滚可在一条命令内完成，回滚后三服务 `healthy`、健康接口 `ready=true`；
- 回滚不触发重新构建（`--no-build`）。

**不作主张**

- 不是多副本、滚动升级或故障自愈；单机 Compose，容器重启由 `restart: unless-stopped` 提供，不是编排层的高可用；
- 未做真实流量压测，未验证并发与容量；
- 未接入生产密钥，演练使用演示配置；
- 回滚目标来自本地已构建镜像，不涉及镜像仓库与签名校验。
