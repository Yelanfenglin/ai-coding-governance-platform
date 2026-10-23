# 本地 RAG Milvus

从 `apps/rag-api` 启动：

```powershell
docker compose -f infra/milvus/compose.yaml up -d
$env:PYTHONNOUSERSITE='1'
.venv/Scripts/python.exe -B scripts/serve.py
```

Milvus 继续使用固定镜像 `v2.6.21`、本机端口 `19531` 和已有的 `rag-milvus-data`。
etcd 使用与原镜像内嵌组件相同的 `v3.5.23`，直接打开原目录
`/var/lib/milvus/etcd`，不生成空库或重置成员。etcd 只在 RAG 的 Docker 网络内提供服务，
不向宿主发布端口。Milvus 保留 local storage，没有新增对象存储服务。

部署结构参考 [Milvus 官方 standalone compose](https://github.com/milvus-io/milvus/blob/v2.6.21/deployments/docker/standalone/docker-compose.yml)。
本地适配是复用既有数据目录与镜像版本，并用 `depends_on: service_healthy` 等待 etcd
可提交请求后再启动 Milvus。

2026-10-23 的冷启动故障出现在内嵌 etcd 的选主阶段：`InitEtcdServer` 启动 etcd 后没有
等待 ReadyNotify，服务初始化的 ID 分配收到 `etcdserver: leader changed` 就 panic。
依据为 [嵌入式启动源码](https://github.com/milvus-io/milvus/blob/v2.6.21/pkg/util/etcd/etcd_server.go)
与 [Session.Init/getServerID](https://github.com/milvus-io/milvus/blob/v2.6.21/internal/util/sessionutil/session_util.go)。
这不代表数据卷损坏；修复针对组件就绪顺序，没有改动检索实现或重新 embedding。

检查：

```text
docker compose -f infra/milvus/compose.yaml ps
docker exec rag-etcd etcdctl --endpoints=http://127.0.0.1:2379 endpoint health
docker inspect --format "{{.RestartCount}}" rag-milvus
```

RAG 的 `/api/rag/health` 还会检查 PostgreSQL、collection、维度和 provider 配置，
服务端口可连接不等于整个 RAG 已就绪。真实检索与工程任务是后续验收层。

修改部署和恢复数据前先停 Milvus 与 etcd，并备份完整数据卷。历史
`embed-etcd.yaml` 保留作配置回退依据；回退到内嵌模式前必须先停止独立 etcd，不能让两个
etcd 进程同时打开这个成员目录。不要用 `down -v`、删除数据目录或空库重建来处理启动错误。
