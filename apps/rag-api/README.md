# rag-api

产品规格书 RAG 的独立服务目录。目前完成的是第一阶段：确定性的 PDF 文档解析与结构重建。检索、向量入库和问答 API 仍以 `../../docs/rag-knowledge-base-design.md` 为设计边界，不在本阶段伪装成已实现。

## 当前目录

```text
app/parser/                 PDF 原生层 + DocLayout anchor + 结构后处理
app/chunking/               语义单元重建 + structure-aware chunk 生成（loader/units/chunker/metadata）
scripts/pp_doclayout_all.py PP-DocLayoutV3 批处理与断点续跑
scripts/audit_reconstruction.py 解析产物不变量审计
scripts/audit_chunks.py     chunk 统计与不变量验收（Step 4）
tests/test_parser.py        单元测试与可选的本地 PDF 回归
tests/test_chunking.py      chunk 层单元测试
docs/chunk-strategy.md      chunk 分块规划（v1 基线）
docs/parser-pipeline.md     完整解析流程与处理技巧
docs/rag-experience.md      RAG 链路经验库：文档解析问题、根因与解决方案
docs/parser-visual-audit.md 表格/快照人工视觉审计基线
artifacts/                  本地版面模型产物，不提交
storage/parsed/             本地解析结果，不提交
storage/chunks/             chunk 层产物（chunks.jsonl），不提交
```

旧版 `parser_v0` 和 `parser_v1_minimal` 已收敛为唯一入口 `app.parser`。解析代码不依赖 PostgreSQL、Milvus 或外部 LLM。

Phase 2 的 PostgreSQL、Milvus、embedding、chunking 和 API 只保留在设计文档中。当前不提交无调用者、无迁移、无测试的半成品脚手架；进入 Phase 2 时按已确认契约完整落地。

## 解析流程

1. `pp_doclayout_all.py` 生成页面 anchor，负责大致区域和阅读顺序。
2. `app.parser` 从 PDF 原生 text/span/vector 层恢复正文、表格和图片快照。
3. 失败的 table anchor 只在原生分隔线形成完整二维记录时升级；缺失的向量图按图元连通性和局部文字密度保守补提。
4. 图片裁剪框由原生图元、同图标签和相邻视觉区域共同校正；已识别正文/图题仍保留为 Markdown 文本。
5. 对外层 cell 内跨列重复对齐的原生物理行，保守恢复无边框嵌套子表；原始 outer topology 与逐行来源同时保留。
6. 跨页表格拼接：上一页以表格结束、下一页第一个正文块就是表格、表头（去空白归一化）一致且列数与水平 span 一致时，把同一张逻辑表的分页片段物理合并为一张表，删除下一页重复表头，输出 `<!-- pdf-pages: N-M -->` 并保留 `source_pages`。不做 rowspan/单元格被页切断的拼接，不合并表格截图。
7. 目录区域重建：含 `目录` 标题或 ≥5 条“点线+末尾页码”视觉行的页面进入 TOC 模式，停止段落重排，按 PDF 视觉基线一行一项重建；末尾页码视为一条目录项的硬闭合。
8. 结构后处理恢复编号标题层级并合并块内视觉换行。
9. `audit_reconstruction.py` 检查文本唯一 ownership、表格边界、语义行来源、图片引用和页序。

表格先使用旧版高置信边线冻结基线结果；只有基线失败且增强后的原生 separator 拓扑仍能通过重复行列校验时，才从 snapshot 升级为结构化表。拓扑优先使用 PDF 原生边线：细填充矩形可恢复为 rule，同一边界上的分段 rule 先做区间并集。文字只作为保守兜底，必须同时满足高密度记录锚点和邻行/邻列 separator 一致性；两行歧义、多行表头和纵排分组标签保持原跨度。若 PDF 将多个视觉单元格编码为一个 text span，解析器仅在字符坐标落入多个独立 cell origin 时按格分配；真实 rowspan/colspan 覆盖的原子格共享同一 origin，因此保持原样。

表内无边框子记录不交给 chunker 二次猜测。解析器保留 outer `rows/cells` 作为保真视图，仅当多个相邻正文行在所有列中具有相同物理行数、跨列基线对齐、行宽留白充分且该模式重复出现时，才额外生成规范化 `semantic_rows`。单个多行 cell、表头换行和普通段落折行保持原样；规则不读取标题词义、引脚名或任何芯片字段。

视觉区域不再直接照搬轻量模型 bbox。孤立图可向原生图元和同带标签扩展；并排或上下相邻的图以 anchor 中线隔离；已有正文/图题 anchor 构成边界，避免正文被截图吞掉。模型完全漏掉的向量图还需同时满足多图元连通、二维尺寸、局部文字密度和已有排他区域不重叠，生产规则不读取文档名、页码、标题词义或芯片字段。

## 文档

- [Chunk 分块规划（v1 基线）](docs/chunk-strategy.md) —— 语义单元模型、按内容类型的切块规则、语义 overlap、metadata 三级字段与验收统计。
- [解析管线：完整流程与处理技巧](docs/parser-pipeline.md) —— 各阶段的职责、关键技术与“几何优先、词表禁用”的泛化约束。
- [RAG 链路经验（当前：文档解析）](docs/rag-experience.md) —— 双栏顺序、rowspan 误吞整列、表格/图分类误判、字符级跨格分配、跨页拼接、目录重建等 20 余个问题的复盘与一句话对策速查；后续 chunking、检索等阶段经验将陆续补入。
- [PDF 表格视觉审计基线](docs/parser-visual-audit.md) —— 人工逐页核对的表格/快照真值计数与逐格 A/B 结果。

## 运行

```powershell
cd apps/rag-api

python scripts/pp_doclayout_all.py `
  --pdf "C:\path\PRODUCT.pdf" `
  --output "artifacts\paddle_doclayout_v3\PRODUCT_doclayout_v3_regions.json" `
  --summary "artifacts\paddle_doclayout_v3\PRODUCT_doclayout_v3_summary.md" `
  --resume

python -m app.parser.parse `
  --pdf "C:\path\PRODUCT.pdf" `
  --layout "artifacts\paddle_doclayout_v3\PRODUCT_doclayout_v3_regions.json" `
  --out "storage\parsed\PRODUCT" `
  --dpi 160

python scripts/audit_reconstruction.py storage\parsed\PRODUCT
python -m pytest -q
```

版面模型依赖是可选项；只重跑已有 layout JSON 时不需要加载 PaddleOCR。依赖分组见 `pyproject.toml`。

## 本地三文档回归

| 文档 | 页数 | page-local 结构化表 | 跨页拼接 | 合并后表块 | 规范化 subrows | 剩余 table snapshot | 结构审计 |
|---|---:|---:|---:|---:|---:|---:|---|
| M8P6272 V1.10 | 79 | 87 | 3 | 84 | 48 | 11 | 通过 |
| M8P5170 V1.06 | 51 | 40 | 2 | 38 | 24 | 7 | 通过 |
| M9F6820 V1.07 | 137 | 147 | 4 | 142 | 78 | 18 | 通过 |

三份结果均为 0 orphan text line、0 duplicate ownership、0 missing snapshot。修复前的 86 个 table snapshot 候选已经逐张人工视觉复核：其中 50 张真表全部升级，剩余 36 张均为图、波形、封装或代码等误标。最终 274 张 page-local 结构化表中，原有 224 张的完整 cell 拓扑与基线逐张一致；跨页拼接后合计 264 个表块（142/38/84），逐格 A/B 除单元格文字的无损重分配外保持不变。新增语义视图恢复 38 个 outer row 内的 150 条对齐子记录，outer topology 仍逐格不变。详见 [PDF 表格视觉审计基线](docs/parser-visual-audit.md)。后续阶段边界见 [RAG 知识库系统设计](../../docs/rag-knowledge-base-design.md)。PDF、layout 产物、解析结果和真实 `.env` 都是本地数据，不应提交。
