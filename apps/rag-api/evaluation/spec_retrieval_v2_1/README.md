# 规格书检索开发题库 v2.1

日期：2026-10-19。对象：已入库的 `spec-chunks-preembed-reviewed-20261018`，`text-embedding-v3`、1024 维；16 份规格书、2,867 个可检索 chunk。

## 与 v2.0 的差异

v2.1 **不修改题目、金标、锚点、定位映射和语料**；它把判分规则从"每次 run 的临时文件"提升为题库的一部分，并修正了复核发现的问题：

1. **规则随题库版本化**：`evidence_rules.json` 进入题库目录，有 `schema_version`、`bank_version`、`reviewer`、`changes_from_v2_0`；打分脚本 `score_results.py` 只从本目录读取规则，不再依赖 run 目录里的临时规则。
2. **去掉硬编码 false**：v2.0 run 的 `manual_incomplete` 把 R034/R038/R045/R048 在**任何 stage** 判 false。v2.1 把这 4 组改为可复算的事实子句，并登记 `pending_description_update`：说明修好后它们会自动生效，不需要再改代码。
3. **补 R036 的等价写法**：`百分比的物理含义未在图中解释` 实际只出现在查询时追加的 `figure_uncertainties` 里（`RagQueryService._assemble_hit()`），v2.1 补充等价表述，并把该依赖登记在 `metadata_dependent`。
4. **拆分区分度高的长字面**：R037-E1、R040-E1 的时钟树分支，R052-E2 的 EEPROM 写入注意事项，R053-E2 的 Flash 缓存器说明由捆绑长句拆为原子子句；寄存器位定义和表格行**保持整体**，避免拆出通用句造成误判。
5. **新增规则级校验**：`validate_bank.py` 除原有静态校验外，还会检查 71 组证据是否被 `facts`/code 精确覆盖、每条子句能否由映射金标块的"送达文本"满足、`pending_description_update` 是否确实仍缺失、代码是否连续、以及同文档内出现 ≥3 次的通用子句告警。

## 文件

| 文件 | 内容 |
|---|---|
| `questions.jsonl` | 64 题（56 正证据 + 8 边界），与 v2.0 相同 |
| `queries.jsonl` / `queries_unscoped.jsonl` | 64 / 54 条接口输入，与 v2.0 相同 |
| `corpus.json` / `current_chunk_mapping.jsonl` | 16 份来源身份与 71 组证据的 chunk 映射，与 v2.0 相同 |
| `evidence_rules.json` | **v2.1 判分规则**（71 组证据：55 组事实 + 16 组代码） |
| `judgment_template.jsonl` | 逐题判定模板 |
| `score_results.py` | 用本目录规则给任意冻结 run 打分（只读，无 API） |
| `validate_bank.py` | 静态 + 规则级校验（可加 `--pdf-root` 复核原 PDF） |
| `manifest.json` | 文件哈希、统计、版本 |

## 判分口径

- 一组 `evidence_requirement` = 一个独立事实要求；规则里每个 evidence 是一组**子句的 AND**，每个子句给若干**等价字面写法（OR）**。
- 匹配是去空白后的字面子串；不接受同义改写。只有同一 `allowed_sources`（产品 + API 版本完全相等）的返回块才算数。
- 代码类证据要求 `required_code_text` 在**同一个块**里连续、按序出现。
- 图片类证据按"实际送达文本"判分；读取到 `image_ref` 不等于模型看见图片。
- **可达上限**：4 组 `pending_description_update` 在说明修正前仍会失败，因此当前 all-required 上限仍是 **52/56**、覆盖组上限 **67/71**；报告指标时同时给出 56 和 52 两个分母。

## 复跑

```powershell
cd apps/rag-api
$env:PYTHONNOUSERSITE='1'
# 题库静态 + 规则校验（不调用模型）
python -B evaluation/spec_retrieval_v2_1/validate_bank.py

# 给一个已冻结的 run 重新打分（不重新检索）
python -B evaluation/spec_retrieval_v2_1/score_results.py `
  --run-dir output/retrieval_top5_no_rerank_20261019 `
  --out-dir output/rules_v2_1_rescore_20261019/top5
```

## 已知边界

- 题目与金标由助手设计，`manifest.json` 的 `independent_engineer_review` 仍为 `pending`。
- 结果只覆盖 16 份规格书的开发集，不外推。
- 规则修改必须来自原 PDF/原图，不能依据某次检索结果反推；修改后需重跑基线并保留旧版本记录。
- v2.0 题库已按仓库约定归档到仓库外并从仓库移除；v2.0 时期的 `output/retrieval_baseline_no_rerank_20261019/` 是冻结记录，复跑需要归档的 v2.0 文件。
