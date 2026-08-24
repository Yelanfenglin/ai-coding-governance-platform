# 产品规格书 RAG 知识库系统设计文档

- 项目：`ai-coding-governance-platform`（fork 自 Yelanfenglin/ai-coding-governance-platform）
- 目标模块：`apps/rag-api`（独立服务）
- 作者 / 评审：RAG 开发
- 状态：**Phase 1 文档解析已实现并完成三文档回归；Phase 2-4 待实现**
- 日期：2026-08-19

> 本文档在动手写代码之前定义系统边界、数据模型、解析管线、检索方案与接口契约。评审通过后按此实现。

当前实现边界与运行命令以 `apps/rag-api/README.md` 为准。本文第 5-8 节仍是后续入库、检索与 API 的目标设计，不代表代码已经落地。

---

## 1. 背景与目标

企业需要把**各种产品规格书（Datasheet / SPEC）** 沉淀为可检索、可问答的知识库。业务对象是**芯片 / MCU 等硬件产品的规格书**（例：`M8P6272_SPEC_V1.10CN.pdf`，79 页，8-bit MTP MCU，含引脚图、寄存器定义表、波形图、框图）。

知识库被 **agent 运行时**调用（agent 在写代码 / 排查问题时查询"某个寄存器的配置""某个引脚的功能"）。

### 1.1 核心诉求
1. **入库是"冷启动 / 触发式"批处理**，不是每次启动解析。上传一份新产品规格书时跑一次，结果固化进向量库 + 元数据库。
2. **运行时查询是实时轻量**的：question → embedding → 向量检索 → rerank → LLM 生成，常驻服务。
3. **支持增量更新**：
   - 上传**新产品**规格书 → 新增入库；
   - 上传**旧产品的新版本**规格书 → 替换重建该产品的索引，存档旧版本。
4. 解析质量优先：**跨页表格、引脚图、波形图、框图**是检索重点，纯文本抽取不够，需版面分析 + 结构化恢复。

### 1.2 明确不做（第一版）
- 不做通用文档管理后台 UI（只提供 API）。
- 不做自然语言语义去重的复杂审计（只做版本级别的替换）。
- 图形深度语义理解（波形含义、框图导通关系）由视觉模块提供"图区域 + 图上文字"，**VLM 语义描述作为增强字段（二期可选）**。

---

## 2. 环境事实与关键决策

| 项 | 结论 | 依据（已核实） |
|---|---|---|
| 仓库位置 | `D:\code\ai-coding-governance-platform`（正式 clone） | 用户指定拉取到 `D:\code` |
| RAG 代码 | **新建独立服务 `apps/rag-api`**，不并入 `apps/api-next` | 与 Codex 控制面解耦，独立扩缩/配置 |
| 解析引擎 | **本地 PaddleOCR**：PP-DocLayout-S（版面）+ PP-OCRv6_small（det/rec）+ 方向分类 | 环境已装 `paddleocr 3.7 / paddlepaddle 3.3 / paddlex 3.7`，权重已下载（见下） |
| PP 权重配置 | 用独立 `RAG_PADDLE_CACHE` 指向本地缓存；仓库默认 `apps/rag-api/artifacts/paddlex` | 路径可覆盖，不依赖其他项目临时目录 |
| 解析代码 | 独立 `apps/rag-api/app/parser/` | 不运行时导入或修改 stockpilot |
| 向量库 | Phase 2 使用独立 `rag-milvus`，宿主端口 `19531` | 不复用 stockpilot 容器或数据卷；编排文件随完整入库层一起落地 |
| 元数据库 | **Windows 本地原生 PostgreSQL 18**（`D:\PostgreSQL\18`，端口 `5432`，服务运行中） | **建独立库 `rag_kb`，不共用任何现有库** |
| 数据隔离原则 | RAG 数据**绝不写入** stockpilot / 其他项目的库 与 Milvus collection | 用户明确要求 |
| embedding / rerank / LLM | **HuggingFace API**：BGE-M3（embedding）、BGE-reranker（重排）、Qwen 系列 / DeepSeek（生成） | Phase 2 使用独立 `RAG_HF_TOKEN`，禁止从其他项目读取凭据 |
| Paddle OCR 降级 | 本地 PaddleOCR 为首选；云端 OCR failover 为后续可选能力 | 尚未接入当前解析入口 |

### 2.1 数据隔离原则（重要）
- **PostgreSQL**：仅使用 `rag_kb` 这一个库（schema `public` 或 `rag`）。连接凭证独立，绝不读写 stockpilot / 其他库。
- **Milvus**：新建独立 collection（默认 `rag_specs`），与 stockpilot 的 `stockpilot_quality_policies` 完全分开。检索/写入全用独立 Collection。
- **配置**：所有密钥/连接串写在 `apps/rag-api/.env`（该文件进 `.gitignore`，禁止提交）。提供 `.env.example` 模板。
- 设计上假定部署方提供：本地 PG18 的可用账号密码 + `rag_kb` 库（或 RAG 服务启动时自动建库）。

---

## 3. 总体架构

```
                    ┌────────────────────────────────────────────────┐
                    │              apps/rag-api (FastAPI)            │
                    │                                                │
  上传 PDF ───────▶ │  POST /api/rag/documents   (multipart)        │
 (新产品/新版本)     │        │                                       │
                    │        ▼                                       │
                    │  ┌──────────────── 后台任务  (解析入库批处理)   │
                    │  │  ① Ingestion: 文档/版本登记                  │
                    │  │  ② Parse: 版面分析+OCR+表格+图裁剪           │
                    │  │  ③ Chunking: 结构化切块                      │
                    │  │  ④ Embed: BGE-M3 (HF API)                   │
                    │  │  ⑤ Store: PG元数据 + Milvus向量              │
                    │  │  ⑥ 版本替换/存档                             │
                    │  └──────────────────────────────────────────────┘
                    │                                                │
  查询 ─────────── ▶│  POST /api/rag/query  (问答带引用)             │
                   │   GET  /api/rag/search (仅检索返回chunks)      │
                    │        │                                       │
                    │        ▼                                       │
                    │  RAG Query: 问题→embed→Milvus topK→            │
                    │   BGE-rerank→拼上下文→HF LLM 生成→带引用       │
                    └──────┬──────────────────────┬──────────────────┘
                           │                      │
                    ┌──────▼──────┐        ┌──────▼──────┐
                    │ PostgreSQL   │        │ Milvus      │
                    │ rag_kb       │        │ collection  │
                    │ 文档/版本/chun│        │ rag_specs   │
                    │ k 元数据     │        │ 向量索引     │
                    └─────────────┘        └─────────────┘
```

运行形态：**一个常驻 FastAPI 进程**。入库是可在同一进程内经后台任务执行的批处理，也可通过独立 CLI ingest 命令触发（推荐提供 CLI，便于手动批量灌库，不占用查询服务）。

---

## 4. 解析管线（Phase 1 已实现）

### 4.1 当前实现

| 模块 | 职责 |
|---|---|
| `scripts/pp_doclayout_all.py` | PP-DocLayoutV3 页面 anchor 批处理、断点续跑、原子 checkpoint |
| `app/parser/native_pdf.py` | PyMuPDF 原生 block/line/span/vector/image 提取 |
| `app/parser/anchor.py` | 每条原生文字行绑定唯一 layout owner，漏检行生成保守 text anchor |
| `app/parser/table.py` | ruled-grid、原子单元格、rowspan/colspan 与保守 topology repair |
| `app/parser/page.py` | PageIR、双栏阅读顺序、排他 ownership 与 snapshot 兜底 |
| `app/parser/postprocess.py` | 编号标题恢复、负向 note 过滤、块内语义换行 |
| `app/parser/document.py` | 章节路径与跨页表格显式链接 |
| `app/parser/renderer.py` | Markdown/HTML 表格、图片引用和注释 |

旧的 `parser_v0` 与 `parser_v1_minimal` 已删除；正式入口唯一为 `python -m app.parser.parse`。

### 4.2 分析流程

1. **版面 anchor**：离线运行 PP-DocLayoutV3，得到 bbox、label 和阅读顺序。
2. **原生提取**：PyMuPDF 提取 text/span/vector/image；不重新按 y 坐标拼接原生行。
3. **唯一 ownership**：TABLE/VISUAL/NOISE 排他拥有区域文字，未命中的原生行进入自动 text anchor。
4. **结构恢复**：真实 vector rule 形成 ruled table；无法可靠恢复的表或图保留截图与原生文字 annotations。
5. **保守后处理**：表格分类先冻结，再修复错误 merge；编号标题按深度建树；同块物理换行转为语义换行。
6. **全文产物**：输出单页 JSON、全文 `pages.json`、`full.md`、snapshot 与质量摘要。

### 4.3 本阶段边界

- 已验证原生文字层规格书；扫描 PDF 的 PP-OCRv6 文字识别尚未接入正式入口。
- 表格跨页只写显式 continuation metadata，不物理拼接 rows。
- 图片保留截图与 annotations；VLM caption、图关系理解尚未实现。
- chunking、embedding、PostgreSQL/Milvus 入库属于 Phase 2，不在解析器中混入临时实现。
- 页眉页脚噪点由 layout NOISE anchor 与页边兜底剔除，不进正文。

---

## 5. 数据模型

### 5.1 PostgreSQL `rag_kb`（关系 / 元数据）

命名规范：`snake_case`，主键 UUID，时间戳 `timestamptz`。用 SQLAlchemy 2 async + Alembic。

**表：`documents`**（一份规格书文件 = 一个版本记录）
| 列 | 类型 | 说明 |
|---|---|---|
| id | UUID PK | |
| product | varchar | 产品型号，如 `M8P6272` |
| product_family | varchar? | 可选产品系列 |
| version | varchar | 规格书版本，如 `V1.10` |
| language | varchar | `zh` / `en` / `zh-CN` |
| title | varchar | 文档标题 |
| filename | varchar | 原始文件名 |
| source_url | varchar? | 来源地址 |
| file_sha256 | varchar(64) | 去重 / 幂等 |
| status | enum | `pending / parsing / parsed / failed / superseded` |
| current | bool | 是否该产品当前有效版本 |
| stored_by | varchar? | 上传者 |
| created_at / updated_at / superseded_at | timestamptz | |
| parse_error | text? | 失败原因 |

唯一约束：`(product, version)`；`current` 上建部分唯一索引保证每产品只有一个 current=1。

**表：`chunks`**（解析出的片段，和向量一一对应）
| 列 | 类型 | 说明 |
|---|---|---|
| id | UUID PK | |
| document_id | FK documents.id | |
| chunk_index | int | 文档内序号 |
| kind | enum(`TEXT/TABLE/VISUAL/MIXED`) | KnowledgeChunkKind |
| section_path | varchar | 章节路径 `1 > 1.1 > 1.1.2` |
| section_title | varchar | |
| page_start / page_end | int | |
| text | text | chunk 正文（markdown for table） |
| table_repr | jsonb? | 若为表格，存结构化 {headers,rows} |
| image_ref | varchar? | 若为图，指向存储的图 resource |
| embedding_id | varchar | Milvus 主键（`{doc}:{idx}` 之类） |
| created_at | timestamptz | |

索引：`document_id`，`(document_id, chunk_index)`，`(kind)`。

**表：`milvus_filters`（可选扁平 metadata）** 或在 Milvus 内用 scalar 字段即可，不单独建表。详见 5.2。

### 5.2 Milvus `rag_specs`（向量）

服务端：`http://127.0.0.1:19531`；dimension 由 BGE-M3 决定（1024）。

字段（collection schema）：
| 字段 | 类型 | 说明 |
|---|---|---|
| id | VarChar(128) PK | `{doc_uuid}:{chunk_index}` |
| vector | FloatVector(1024) | BGE-M3 中文/英文语义向量 |
| document_id | VarChar(64) | 用于 filter |
| product | VarChar(64) | filter |
| version | VarChar(64) | filter |
| kind | VarChar(16) | TEXT/TABLE/VISUAL |
| chunk_index | Int64 | |
| page_start / page_end | Int64 | |
| section_path | VarChar(256) | 可用于展示溯源 |
| text | VarChar(4000) | chunk 原文（检索后展示给 LLM/用户） |

索引：IVF_FLAT 或 HNSW 的余弦相似度（**HNSW, cosine** 对 RAG 召回效果好）。`create_index`、dynamic=false。Collection 和 索引通过 `pymilvus 2.6.17` 管理。

**检索过滤**：按 `product` / `version` / `kind` 做 Filter（Milvus 标量过滤），例如"只查 M8P6272 且 TABLE 类型"。

### 5.3 图片资源存储
图上 `image_ref` 可指向 `apps/rag-api/storage/images/<doc_id>/<chunk_id>.png`（本地磁盘，启动时配置 `STORAGE_DIR`）。第一版不做对象存储。禁止提交。

---

## 6. 版本管理（新产品 vs 新版本）

| 场景 | 行为 |
|---|---|
| 上传新品 `M8P6272 V1.10`（无该 product） | 新建 `documents` 记录（current=1），解析入库，`chunks` + Milvus 写入 |
| 上传已有 product 的**新版本**（如 `V2.00`） | 新建版本记录并设为 current=1；**旧版本标记 superseded**（current=0，索引可从查询默认过滤掉，或按需保留可查）；删除旧版本在 Milvus 中向量（或保留+filter current） |
| 上传**相同 product 相同 version**（幂等/覆盖） | 以 `file_sha256` 判重：若一致→返回"已存在"；若不同→按版本替换处理 |

- 默认查询只面向 `current=1` 的文档；如需历史对比，显式传 `version` 过滤。
- 为降低实现复杂度：**新版本入库成功后，才把旧版本 superseded**（避免中途失败丢可用数据）。失败时新版本标记 `failed`，旧版本保持 current。

---

## 7. 检索与生成（RAG 问答）

### 7.1 检索流程
```
question
  → EmbeddingProvider.embed([question])        # HF BGE-M3 feature_extraction
  → Milvus search(top_k=8, filter=current & [product])   # 余弦
  → Reranker: BGE-reranker (HF sentence_similarity/rerank) 对 question×命中重排
  → 取 top_n(context_window)
  → 拼接上下文：带 [source: 产品/版本/章节/页码/kind] 标注
  → LLM 生成（HF chat）→ 返回 answer + citations
```

- **Embedding 接口抽象**：`EmbeddingProvider = protocol`，提供两个实现：
  - `HFEmbeddingProvider`：调用 HuggingFace `feature_extraction`（BGE-M3），dimension 1024。
  - `DeterministicHashEmbedding`（Phase 2 在本仓库独立实现，可重放开发 / 离线测试，**不用于生产**）。
- **Reranker**：封装 HF `sentence_similarity`（BGE-reranker-base/v2m）。
- **LLM**：用 HF `chat_completion` 免费模型（如 `Qwen/Qwen2.5-7B-Instruct`、`meta-llama/Llama-3.1-8B-Instruct` 或组织授权的 DeepSeek HF 端点）。模型名可配置。
- **超时/重试**：HF serverless 免费额度有限，调用封装重试 + 超时 + rate-limit 退避；可配置降级（如只检索不生成）。

### 7.2 答案格式（citations）
`POST /api/rag/query` 返回：
```json
{
  "answer": "...",
  "citations": [
    {"product":"M8P6272","version":"V1.10","section_path":"2.2.6","page":15,
     "kind":"TABLE","text":"...","score":0.87}
  ],
  "model":"qwen/...-instruct",
  "latency_ms": 1230
}
```

---

## 8. API 契约（FastAPI）

根路径统一 `RAG_API_PREFIX=/api/rag`。鉴权第一版为**可选共享 API Key**（`X-API-Key`，配置项），不强绑现有控制面登录。

| 方法 & 路径 | 说明 |
|---|---|
| `POST /api/rag/documents` | 上传规格书 PDF（multipart：`file` + 表单 `product` `version` `language?`）。创建解析任务，返回 `{job_id}` |
| `GET  /api/rag/documents/{doc_id}/status` | 查询解析状态（pending/parsing/parsed/failed）与进度 |
| `GET  /api/rag/documents` | 列出文档（过滤 product/version/current） |
| `DELETE /api/rag/documents/{doc_id}` | 删除某版本及其向量（可选） |
| `POST /api/rag/query` | 问答（question, product?, version?, top_k?）→ answer + citations |
| `GET  /api/rag/search` | 仅检索（返回 chunks，不生成） |
| `GET  /api/rag/documents/{doc_id}/context/{chunk_id}` | 溯源：某 chunk 的原文/表格/图 |
| `GET  /api/rag/health` | 探活（可选同时反映 Milvus / PG / HF 连通） |

**同步入库存档（可选 CLI）**：
```
python -m rag_api.ingest run --pdf M8P6272.pdf --product M8P6272 --version V1.10
```
用于批量灌库，不占 HTTP 常驻进程。

---

## 9. 目标目录结构（Phase 2-4，尚未实现）

当前 Phase 1 实装目录见 `apps/rag-api/README.md`。下列结构是入库、检索与 API 完成后的目标，不是当前文件清单。

```
apps/rag-api/
├─ pyproject.toml            # deps: fastapi, uvicorn, sqlalchemy[asyncio], asyncpg|psycopg,
│                            # pydantic-settings, alembic, pymilvus, paddleocr, pymupdf,
│                            # huggingface_hub, httpx, python-multipart
├─ alembic.ini / alembic/    # migrations (documents, chunks)
├─ .env.example              # 模板（无真实密钥）
├─ .env                      # 本地（git-ignored）
├─ app/
│  ├─ main.py                # FastAPI 入口，路由注册，lifespan 连接 Milvus/PG/关闭 Providers
│  ├─ settings.py            # Settings(pydantic-settings, env_prefix=RAG_)
│  ├─ db/
│  │  ├─ models.py           # documents / chunks 表
│  │  ├─ session.py          # async engine/SessionLocal（连 rag_kb）
│  │  ├─ schema.py           # 建库/连接探活
│  │  └─ repo.py             # Postgres 存取（文档/版本/chunk）
│  ├─ schemas.py             # REST 请求/响应模型
│  ├─ routers/
│  │  ├─ documents.py        # 上传/状态/列表/删除
│  │  ├─ query.py            # /query /search /context
│  │  └─ health.py
│  ├─ model_ai/
│  │  ├─ embeddings.py       # Provider 协议 + HF(BGE-M3) + Deterministic
│  │  ├─ reranker.py         # HF BGE-reranker 封装
│  │  └─ llm.py              # HF chat_completion 封装（含重试/降级）
│  ├─ vectordb/
│  │  └─ milvus_store.py     # collection 管理 + insert + search + delete + filter
│  ├─ ingest/
│  │  ├─ pipeline.py         # 串联: ingest→parse→chunk→embed→store→版本替换
│  │  ├─ tasks.py            # 后台任务执行器（持久化 job 状态）
│  │  └─ cli.py              # python -m rag_api.ingest run ...
│  ├─ parser/                # Phase 1 已实现的确定性解析器（保持唯一入口）
│  │  ├─ native_pdf.py / anchor.py / page.py
│  │  ├─ table.py / native_tables.py / postprocess.py
│  │  └─ document.py / renderer.py / parse.py
│  ├─ chunking/              # Phase 2：PageIR -> 可检索规格书 chunks
│  └─ storage.py             # 图片资源落盘路径管理
└─ tests/
   ├─ test_chunking.py / test_version_replace.py / test_milvus.py
```

依赖组（在 pyproject 里分层）：
- `base`：fastapi, uvicorn, pydantic-settings, sqlalchemy[asyncio], alembic, asyncpg, python-multipart, httpx, huggingface_hub, pymilvus, pymupdf/fitz
- `layout`：paddleocr, paddlepaddle（rag-api 独立可选 group）
- `dev`：pytest, pytest-asyncio, ruff

---

## 10. 配置（.env / Settings）

`Settings`（`env_prefix=RAG_`），从 `.env` 读取：
```
# PGRAG
RAG_DATABASE_URL=postgresql+asyncpg://user:pass@127.0.0.1:5432/rag_kb
# Milvus
RAG_MILVUS_URI=http://127.0.0.1:19531
RAG_MILVUS_COLLECTION=rag_specs
# HF
RAG_HF_TOKEN=hf-...
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
RAG_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# Paddle
RAG_PADDLE_CACHE=.../.tmp/paddlex          # 指向已有 official_models 目录
RAG_PADDLE_LAYOUT_MODEL=PP-DocLayout-S
RAG_PADDLE_OCR_TIER=small
# 行为
RAG_STORAGE_DIR=storage
RAG_API_PREFIX=/api/rag
RAG_API_KEY=...                             # 可选鉴权
RAG_OCR_MODE=auto                         # off|auto|required
RAG_LAYOUT_MODE=auto
RAG_EMBEDDING_FALLBACK=deterministic      # 无 HF 时降级
RAG_RENDER_DPI=168
```
`.env` 必须 git-ignore，`.env.example` 提交空值模板。

---

## 11. 与 stockpilot 的边界（合规）

- 当前 `app/parser/` 是独立实现，不从 stockpilot 运行时导入任何代码。
- 后续若参考通用 Provider 协议，必须在本仓库内实现并保留来源/许可记录，不复制制度域模型。
- **绝不修改** `D:\code\stockpilot` 仓库内任何文件。

---

## 12. 分期实施与验收

### Phase 1 — 解析可靠
- [x] 唯一入口 `app.parser`：原生文字、layout anchor、表格、图片 snapshot、章节路径与全文 Markdown。
- [x] M8P6272、M8P5170、M9F6820 三份规格书完成全量回归与结构审计。
- [x] conservative table repair、numbered heading recovery、paragraph reflow 有单元测试和真实页面回归。
- [ ] 扫描 PDF / 无原生文字层的 PP-OCRv6 正式入口与回归样本。
- 验收现状：三份样本均为 0 orphan、0 duplicate ownership、0 missing snapshot；layout/table 分类数量保持旧基线。

### Phase 2 — 数据落库
- [ ] PG `rag_kb` 建库 + Alembic 迁移建 `documents/chunks`。
- [ ] Milvus 建 `rag_specs` collection + HNSW 索引。
- [ ] `ingest.pipeline` 端到端：解析→切块→embed(BGE-M3 / 无 token 时 hash 降级)→写 PG + Milvus。

### Phase 3 — 检索问答
- [ ] `/query /search /context` 接口，BGE-rerank + HF LLM 生成，返回带 citations。
- [ ] 验证检索示例：问"M8P6272 的 ADCON0 寄存器低 2 位是什么"，能命中断言对应表格 chunk 并给出带页码引用。

### Phase 4 — 版本与运维
- [ ] 上传新版本替换、幂等判重、失败回滚逻辑（旧版本保持 current）。
- [ ] CLI ingest、后台任务状态推进、测试 + README。

### 全局验收标准
- 不写入任何非 `rag_kb` PG 库；不写 stockpilot 的 Milvus collection。
- `.env` 不进 git；`AGENTS.md` 的安全红线（不提交秘密/上传内容）遵守。

---

## 13. 风险与开放问题

1. **HF 免费 API 额度/排队**：BGE-M3 embedding、rerank、LLM 均受免费速率限制。缓解：本地 `DeterministicHashEmbedding` 兜底 + 重试退避；生产可换国内 API（如百炼 text-embedding-v3）。
2. **PyMuPDF 依赖**：当前解析器明确依赖 PyMuPDF，不做静默 pypdf 降级；缺依赖应显式失败。版本约束已写入 `apps/rag-api/pyproject.toml`。
3. **Milvus 隔离**：Phase 2 必须使用独立容器、端口和数据卷，并验证 schema、备份与恢复；当前不保留不可运行的半成品编排。
4. **图片语义理解（VLM）**：一期只做"图 + 图上文字 + 所在章节 + 查看原图"，VLM caption 二期再接入 HF 免费 VLM。
5. **登录鉴权**：一期用 `X-API-Key` 简单鉴权，不接现有控制面登录；如需与主平台统一登录，二期再对接。

---

## 14. Phase 2 待办依赖（不阻塞当前解析器）
- [ ] 本地 PG18（`rag_kb`）的连接账号/密码（或确认用 `postgres` 超级用户 + 密码）。
- [ ] HF token 写入 `apps/rag-api/.env`（`RAG_HF_TOKEN`）。
- [ ] 为部署环境设置独立 `RAG_PADDLE_CACHE`，或把批准的权重放入默认 `apps/rag-api/artifacts/paddlex`。
