# Milvus 部署

使用固定版本 Milvus `v2.6.21` 与 etcd `v3.5.23`，通过健康检查按顺序启动。数据卷为 `rag-milvus-data`，Milvus 对宿主使用 `19531` 端口，etcd 仅在 Docker 网络内开放。

在 `apps/rag-api` 下执行：

```powershell
docker compose -f infra/milvus/compose.yaml up -d
docker compose -f infra/milvus/compose.yaml ps
```

etcd 打开数据卷中的 `/var/lib/milvus/etcd`，Milvus 使用同一数据卷与 local storage。RAG 的连接地址、collection 和向量维度在 `.env` 中配置。

## 就绪检查

```powershell
docker exec rag-etcd etcdctl --endpoints=http://127.0.0.1:2379 endpoint health
docker inspect --format "{{.RestartCount}}" rag-milvus
```

再请求 RAG `/api/rag/health`，确认数据库、向量库和 provider 配置就绪。端口可连接不等于整个检索链已就绪。

## 数据维护

更改部署或恢复数据前，先停止 Milvus 和 etcd 并备份完整数据卷。不要通过 `down -v` 或删除数据目录处理启动问题。

`embed-etcd.yaml` 保存内嵌 etcd 配置；切回内嵌模式前停止独立 etcd，避免两个进程同时打开同一成员目录。
