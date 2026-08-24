# rag-api

产品规格书 RAG 的独立服务目录。目前完成的是第一阶段：确定性的 PDF 文档解析与结构重建。检索、向量入库和问答 API 仍以 `../../docs/rag-knowledge-base-design.md` 为设计边界，不在本阶段伪装成已实现。

## 当前目录

```text
app/parser/                 PDF 原生层 + DocLayout anchor + 结构后处理
scripts/pp_doclayout_all.py PP-DocLayoutV3 批处理与断点续跑
scripts/audit_reconstruction.py 解析产物不变量审计
tests/test_parser.py        单元测试与可选的本地 PDF 回归
artifacts/                  本地版面模型产物，不提交
storage/parsed/             本地解析结果，不提交
```

旧版 `parser_v0` 和 `parser_v1_minimal` 已收敛为唯一入口 `app.parser`。解析代码不依赖 PostgreSQL、Milvus 或外部 LLM。

Phase 2 的 PostgreSQL、Milvus、embedding、chunking 和 API 只保留在设计文档中。当前不提交无调用者、无迁移、无测试的半成品脚手架；进入 Phase 2 时按已确认契约完整落地。

## 解析流程

1. `pp_doclayout_all.py` 生成页面 anchor，负责大致区域和阅读顺序。
2. `app.parser` 从 PDF 原生 text/span/vector 层恢复正文、表格和图片快照。
3. 失败的 table anchor 只在原生分隔线形成完整二维记录时升级；缺失的向量图按图元连通性和局部文字密度保守补提。
4. 图片裁剪框由原生图元、同图标签和相邻视觉区域共同校正；已识别正文/图题仍保留为 Markdown 文本。
5. 结构后处理恢复编号标题层级并合并块内视觉换行。
6. `audit_reconstruction.py` 检查文本唯一 ownership、表格边界、图片引用和页序。

表格先使用旧版高置信边线冻结基线结果；只有基线失败且增强后的原生 separator 拓扑仍能通过重复行列校验时，才从 snapshot 升级为结构化表。拓扑优先使用 PDF 原生边线：细填充矩形可恢复为 rule，同一边界上的分段 rule 先做区间并集。文字只作为保守兜底，必须同时满足高密度记录锚点和邻行/邻列 separator 一致性；两行歧义、多行表头和纵排分组标签保持原跨度。

视觉区域不再直接照搬轻量模型 bbox。孤立图可向原生图元和同带标签扩展；并排或上下相邻的图以 anchor 中线隔离；已有正文/图题 anchor 构成边界，避免正文被截图吞掉。模型完全漏掉的向量图还需同时满足多图元连通、二维尺寸、局部文字密度和已有排他区域不重叠，生产规则不读取文档名、页码、标题词义或芯片字段。

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

| 文档 | 页数 | 最终结构化表 | topology 升级 | 剩余 table snapshot | 人工确认真表总数 | 结构审计 |
|---|---:|---:|---:|---:|---:|---|
| M8P6272 V1.10 | 79 | 87 | 17 | 11 | 87 | 通过 |
| M8P5170 V1.06 | 51 | 40 | 4 | 7 | 40 | 通过 |
| M9F6820 V1.07 | 137 | 147 | 29 | 18 | 147 | 通过 |

三份结果均为 0 orphan text line、0 duplicate ownership、0 missing snapshot。修复前的 86 个 table snapshot 候选已经逐张人工视觉复核：其中 50 张真表全部升级，剩余 36 张均为图、波形、封装或代码等误标。最终 274 张结构化表中，原有 224 张的完整 cell 拓扑与基线逐张一致。详见 [PDF 表格视觉审计基线](docs/parser-visual-audit.md)。后续阶段边界见 [RAG 知识库系统设计](../../docs/rag-knowledge-base-design.md)。PDF、layout 产物、解析结果和真实 `.env` 都是本地数据，不应提交。
