# Chunk 分块规划（v1 基线）

> 本文是 `apps/rag-api` chunk 阶段的设计基线。v1 只验证一句话：
> **利用解析出的章节、表格和寄存器结构，能否稳定生成比固定 token 切块更完整的检索单元。**
> 顺序：语义单元边界 → chunk 边界 → embedding_text 自洽。检索增强（重排、引用图、入库）全部后置。

## 1. 定位与原则

- chunk 是解析结果之上的 **retrieval view**，不是唯一结构：
  `Document → Page → Section → Block → Table` 解析树保留不变（adjusting embedding 模型 /
  切块策略只需重新生成 chunk，不需重新解析 PDF）。
- 两级：**Level 1 语义 chunk**（入 embedding/检索）；**Level 2 parent chunk**（命中后扩展，
  不单独入向量，本质是 context range / child container）。
- 结构优先：heading / table / rowspan / page / caption / block type 是强信号，优先于模型。
- 语义完整性 > 目标尺寸；确定性为主，**不用 LLM 切块**。
- `content`（忠实展示）与 `embedding_text`（检索文本）**分开保存**，原始 Markdown 不篡改。
- 生产规则禁用词表；metadata 自结构导出，可审计。

## 2. 总体架构

```text
PDF → Layout parsing → Document reconstruction（已完成）
    → Semantic reconstruction   块 → SemanticUnit           [新增]
    → Chunk generation          SemanticUnit → chunk+meta   [新增]
    → Embedding / RAG（Phase 2）
```

新增模块（不改解析器）：

```text
app/chunking/loader.py    读 storage/parsed/{doc}（pages.json + summary.json）→ ParsedDocument
app/chunking/units.py     SemanticUnit 构建（Text / Register / Table / Figure / Toc）
app/chunking/chunker.py   单元 → chunk（切分 / overlap / content + embedding_text）
app/chunking/metadata.py  ChunkMetadata schema、chunk_id、token 估算
scripts/audit_chunks.py   Step 4 验收：统计 + 不变量
```

## 3. SemanticUnit（v1 五类）

```text
SemanticUnit
├── TextUnit      章节小节（块序列；示例/代码/注释就是普通块，不单独立类）
├── RegisterUnit  标题 + 位表 + Bit 解释 + 注（原子包，允许超尺寸）
├── TableUnit     表题 + 表头 schema + 行（可拆 row group；revision 记录以
│                 semantic_type="revision"，不建独立类）
├── FigureUnit    轻量：image_ref + caption + annotations + section_path + page
└── TocUnit       目录（exclude_from_embedding，仅导航）
```

明确不做专门类：`ExampleUnit / CodeUnit / NoteUnit / RevisionUnit`。
将来真出现“示例代码经常被切坏”再抽 `ExampleUnit`。

**RegisterUnit 判定（结构信号）**：小节含位定义表，或表后文本出现
`^Bit\s*(\d+|\[\d+:\d+\])` 位定义行；标题形式为 “…(控制)寄存器”。
判定为 register 后，**整个小节作为原子包**（标题+表+位说明+注），不切。

**Toc 判定**：首个 heading 归一化为 `目录`；其后至下一个编号 heading 的块均属 TOC。

**FigureUnit**：snapshot 块；caption 取所属 section 标题；不绑定复杂前后文。

## 4. 分块规则（按类型）

| 内容 | 规则 |
|---|---|
| 普通正文 | heading-aware packing；**不跨 heading**；超长按段落（必要时句级）拼到 500–700 再切 |
| 寄存器 | 原子包（允许超 normal_max；极端 >2500 才 fallback 按 bit 组） |
| 大表 | 先按 **col0 rowspan 组**切（`ISP1:3V/5V`、指令类别行组整组不拆），再 pack 到目标；每个子 chunk **重复表题 + 完整表头（schema）** |
| 图 | figure 单包 |
| TOC | `exclude_from_embedding=True`，不入向量 |

**小而完整的 chunk 不合并；小而残缺的 TextUnit 才向邻居 pack。**

## 5. 参数（token）

```text
target_tokens   = 512     初始工程参数（与 embedding 维度无直接匹配关系；
                          由 chunk 分布统计 + retrieval eval 调整）
soft_max_tokens = 800     普通 TextUnit：800 开始找切点
normal_max      = 1200    普通 TextUnit：1200 必须切
soft_min        = 150     软下限；合法的语义完整小 chunk（如 4 行寄存器）不合并

atomic unit（register / 完整示例类）：allow_oversize = True
```

## 6. Overlap：语义 overlap（无固定百分比）

| 类型 | overlap |
|---|---|
| 正文（同章节内跨 chunk） | 前一 chunk 最后 1–2 句（≤60 token）+ section path |
| 表格行组 chunk | 表题 + 完整表头（schema，非数据行重复） |
| 指令表 chunk | 章节 + 列头 + 类别名 |
| 寄存器包 / 图 | 无（原子/单包） |
| 跨章节边界 | 0（绝不跨 heading） |

## 7. 语义关联（不靠 overlap 粘连）

- 层级链：`parent_id / prev_id / next_id / section_path / table_id`；
- breadcrumb 注入 embedding_text：`{product} Version {version}` + `> 章节链`；
- 表格 schema 复制（语义延续，非重复）；
- 寄存器原子包；
- 正文引用**原文保留**，v1 **不解析**（`reference_links` 推迟 v2：extraction → resolution → chunk graph）。

**Parent chunk v1** = 纯容器：`id + title + section_path + child_ids + source_pages`，**不做摘要**。

## 8. Chunk Metadata（三级，v1 不全填）

**v1 必须有**：`doc_id, product, version, chunk_id, parent_id, prev_id, next_id,
section_id, section_path, semantic_type, block_types, page_start, page_end,
source_pages, content, embedding_text, tokens, chars, exclude_from_embedding,
parser_version, chunker_version`

**有才写**：`register, table_id, row_group, instruction_category`

**v2 再写**：`entities, reference_links, related_chunk_ids, image_description,
structured_confidence`

**chunk_id**：`{doc_id}:{section_id}:{unit_type}:{ordinal}`，例如
`m8p6272_v1.10:8.6:register:001`；可读名称 `object_key="PWM1CR"` 与稳定 ID 分离。
跨重新 chunk 需稳定时再升级为 `sha1(doc_id + section_path + unit_anchor)[:16]`。

**embedding_text 构造**（不做语义重述/摘要）：

```text
metadata context（breadcrumb + 类型 + register/table 标识）
+ deterministic structural serialization（位表 → `Bit7 PWM13POL` 列表；表 → header+rows）
+ source content（原始文本，轻度折叠空白）
```

## 9. 分层存储（仅约定，不实现）

PG 全量 metadata + content + embedding_text；向量库（Phase 2）只放紧凑过滤字段
（product/version/semantic_type/table_id/register/chunk_id/parent_id/page…）+ 向量。
过滤字段建索引。v1 只产出 `chunks.jsonl` + 统计。

## 10. 验收（Step 4，v1 核心）

**统计（每份文档）**：chunks / text / register / table / figure 数量；
token 分布 p10/p50/p90/p95/max；oversize atomic / undersize semantic /
cross-heading chunks / broken rowspan groups / orphan table chunks。

**不变量（assert）**：

```text
no chunk crosses heading unless explicit
every table chunk has schema（表头）
every register chunk contains its register table
rowspan group never split
toc chunk.exclude_from_embedding == True
source_pages not empty
parent_id exists
```

**人工抽查 30–50 个 chunk**（10 register / 10 text / 10 大表 / 5 instruction /
5 figure / 5 极短 / 5 最长），四问：
单独可理解？缺紧邻上下文？硬塞两个主题？命中后返回它是否足够有用？

## 11. v1 明确不做的

```text
reference_links / related_chunk_ids / parent 摘要 / entities 提取 /
ExampleUnit·RevisionUnit 专门类 / 复杂 Figure 上下文绑定 /
PG / Milvus / BM25 索引 / VLM 图片描述 / reranker / 检索 API
```

## 12. Embedding 调用（两层，v1 演示用 Ollama）

```text
v1 演示（Step 5）    Ollama 本地 API：/api/embed
                    （模型 batiai/qwen3-embedding，本地已部署，dense only，零下载）
正式/后续           bge-m3（本地 FlagEmbedding，1024 维 dense + sparse 混合）
```

- `EmbeddingProvider` 协议解耦：`embed_dense(texts) -> np.ndarray`；实现可换
  （`OllamaEmbeddingProvider` → `LocalBGEM3Provider`），chunk 与 metadata 不动；
- 配置走 `.env`：`OLLAMA_URL` / `OLLAMA_MODEL`；不硬编码；
- v1 演示 = dense 检索 + metadata 过滤；混合（dense+sparse）等换 bge-m3 时再加；
- 数据不出网（localhost），符合数据隔离。
