# 规格书 RAG 服务

面向 MCU 规格书的结构化解析、混合检索与证据问答服务。Agent 通过 HTTP 获取带型号、版本、页码与片段标识的证据，再按需展开原文。

## 检索链路

```text
PDF → 版面与原生结构重建 → 语义分块 → PostgreSQL / Milvus
问题 + 型号/版本 → 向量召回 + BM25 → RRF → 重排 → 上下文与引用
```

- 解析保留章节、寄存器表、合并单元格、跨页表格及图片定位，避免图表文字重复进入正文。
- 分块以语义单元与结构边界组织，元数据关联产品、版本、章节、页码与源文件。
- 向量检索可结合 BM25、RRF 和 reranker；证据选择与生成分层，引用检查单独执行。
- 入库、检索、问答和 trace 参数由 `app/settings.py` 控制，BM25、重排等按部署配置启用。

## 安装与启动

在本目录使用 Python 3.12 环境：

```powershell
$env:PYTHONNOUSERSITE='1'
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

在 `.env` 配置 PostgreSQL、Milvus、embedding 与生成 provider。准备数据库并启动服务：

```powershell
.venv/Scripts/python.exe -m alembic upgrade head
docker compose -f infra/milvus/compose.yaml up -d
.venv/Scripts/python.exe -B scripts/serve.py
```

向量模型与维度应与所选 collection 一致。PDF、解析产物和向量数据由部署者准备；数据接入使用 `python -m app.ingest --help`，解析使用 `python -m app.parser.parse --help`。

Milvus 配置与数据卷说明见 [部署说明](infra/milvus/README.md)。

## API

默认前缀 `/api/rag`；配置 `RAG_API_KEY` 后请求带 `X-API-Key`。

| 接口 | 用途 |
|---|---|
| `GET /health` | 数据库、向量库与 provider 就绪检查 |
| `GET /search` | 按问题、产品、版本与类型检索 |
| `GET /documents/{doc_id}/context/{chunk_id}` | 展开原始证据与邻近上下文 |
| `POST /query` | 检索、生成、引用与回答检查 |

Agent 的 `RAG_API_URL` 应填写完整前缀，例如 `http://127.0.0.1:8000/api/rag`。

## 评测

冻结题库位于 `evaluation/spec_retrieval_v2_2/`，当前数据版 2.3.1，覆盖 16 份规格书，分 development 与 holdout；必要证据组按事实子句、完整代码与来源范围判分。

```powershell
.venv/Scripts/python.exe -m pytest -q -m "not integration"
.venv/Scripts/python.exe -B evaluation/runner.py validate --pdf-root <原始PDF目录>
.venv/Scripts/python.exe -B evaluation/runner.py score --run-dir <已有检索产物> --split development
.venv/Scripts/python.exe -B evaluation/runner.py regression
```

评分和回归只读取已有产物，不重新调用模型。原文校验需要对应 PDF 与 chunk 文件，回归需要已保存运行；缺失输入返回非零。

2026-10-19 冻结批次：指定型号与版本的 holdout Top-5 全必要证据命中 19/20；无范围检索为 11/17。20 道正证据问答中，模型判题的完整引用支撑为 19/20，另 6 道边界题单独判断。检索命中与答案支撑分别计量，不合并成任务正确率。

## 源码

`app/parser` 与 `app/chunking` 负责离线处理；`app/ingest` 负责入库；`app/retrieval`、`selection`、`generation`、`checks` 分别负责检索、证据选择、生成和检查。数据库迁移位于 `alembic/`，解析与数据审计脚本位于 `scripts/`。
