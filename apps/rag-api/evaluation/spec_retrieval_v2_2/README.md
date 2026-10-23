# 规格书 RAG 题库 v2.3.1（目录 spec_retrieval_v2_2）

这份题库用于检索、证据送达和回答边界的回归评测。覆盖 16 份规格书，题目、金标和规则由助手编写并按原始 PDF/代码截图核对；不是独立专家评测或真实用户任务集。

## 入口和范围

- [题库阅读版](题库.md)：问题、来源页码、必要证据、回答要求与关键错误。
- `questions.jsonl`：完整题目与金标；`queries*.jsonl` 只含 ID 和接口输入，不能把金标交给被测系统。
- `current_chunk_mapping.jsonl`：真实 PostgreSQL chunk UUID；`corpus.json` 固定型号、API 版本、原 PDF 和 chunk 文件。
- `evidence_rules.json`：事实子句 AND、等价写法 OR；代码按完整连续行匹配。
- `manifest.json`：从题目计算的统计和文件指纹；`judgment_template.jsonl` 为逐题判定空模板。

本版共 115 题，其中 107 道正证据题、2 道缺失型号/版本题和 6 道回答边界题；共 143 组必要证据。正证据题包括资料冲突识别，命中两处矛盾原文不代表回答已经正确识别冲突。

## 数据划分

原 v2.1 的 64 题保持不变；12 道已审 pilot 题归入 development；上一轮 10 道 holdout 为 R089–R098；本轮 B 扩充新增 30 题（R099–R128），其中 20 development + 10 holdout。合计 development 95、holdout 20。R106 因与 R101 语义重复已删除，不维持固定题数。旧候选中被拒的 ID 不复用。

预留题涵盖跨页代码、串口发送、TC2 门控、OPA 校准、看门狗配置差异、比较器编码差异、串口控制位差异、SPI 写冲突、串口接收错误和 ADC 时钟计算。它们在选题时未使用检索得分，与开发集没有同 family 或相同金标 chunk；仍不能称为独立盲测。细粒度 family 标签和人工语义复核不能保证所有近义概念绝对隔离。

先在 development 上调参。参数冻结后一次性运行 holdout；若据预留题失败调整实现，应将该批题转为开发用途并另留新题。两部分结果分别报告，不能合并为泛化准确率。当前 holdout 没有新增拒答题，边界能力由开发集回归覆盖。

## 判分口径

- `input.top_k=30` 是候选召回池；主报告看 **candidate@5** 和实际 **context**，不能把扩展上下文的收益写成 top-5 检索收益。
- 每组必要证据全部子句满足才命中，同一条证据不因重复 chunk 多计分。产品和 API 版本必须同时匹配。
- 代码要求完整、连续、按序；跨页例程每一页都是必要证据，缺任一页不算全证据命中。回答中跨页顺序、解释质量另按 `answer_checks` 核查。
- 字面规则衡量证据送达，不接受语义改写；它不直接计算答案正确率。缺失信息题和源文冲突题必须检查生成回答，不能用检索命中替代。
- 图片以送达给模型的文本描述判分；找到图片文件不能声称模型看见图片。此前 4 处描述修复保留在规则历史中，已不列为 pending。
- 部分题或重复 ID 的 run 会被拒绝评分，防止静默缩小分母；结果同时输出 development/holdout 分组。

## 本版纠正

R065 原金标只取特性列表 `1K*6`，而同页简介写 `1K*16`。本版改为引用两处并识别原文冲突。R072 删除不相关的 ADC 延时子问题，集中检查 STOP 与外部中断同时发生时的唤醒限制。R071 将断行的同一事实合为一个子句。

v2.1 题目和既有历史实测未修改。本版运行时检索和答案评测均为 `not_run`；静态校验与评分契约测试不调用模型。B 轮题目生成使用过 DeepSeek（调用记录与 `generation_source` 见 manifest），修复轮未重跑生成。

## 复跑

在 `apps/rag-api` 下使用项目 Python 3.12 环境，设置 `PYTHONNOUSERSITE=1`。以下 `python` 必须指向已激活的项目环境。

```powershell
$env:PYTHONNOUSERSITE='1'
python -B evaluation/spec_retrieval_v2_2/validate_bank.py --pdf-root <原始规格书目录>
python -B -m pytest -q evaluation/spec_retrieval_v2_2/test_scoring_contract.py
python -B evaluation/spec_retrieval_v2_2/score_results.py --run-dir <冻结检索产物目录> --split development
```

评分输入沿用 `run.json`、`scoped_results.jsonl`、`unscoped_results.jsonl`。只跑预留题时用 `queries_holdout.jsonl`，并从 `queries_unscoped.jsonl` 中按相同 ID 筛出无过滤输入，评分传 `--split holdout`。`--split all` 要求两轨完整覆盖对应题目。`contexts` 里的块须有对应 `hits` 元数据；邻块扩展的结果应同时保留这些新增块的来源元数据。

维护题库时按“覆盖缺口 → 原始资料 → 问题与金标 → 静态检查 → 评分反例 → 冻结 → 实测”进行。旧 `output/bank_completion_20261019/build.py` 只覆盖到 v2.2/R098，会把 R099-R128 丢掉，不能用于当前版本。

**当前冻结版的离线恢复入口**：从已保存快照恢复，不重新生成题目、不调用 DeepSeek：

```powershell
$src = "<仓库外快照目录>/spec_retrieval_v2_2_2.3.1"
Copy-Item "$src\*" "evaluation/spec_retrieval_v2_2" -Recurse -Force
python -B evaluation/spec_retrieval_v2_2/validate_bank.py --pdf-root <原始规格书目录>
```

快照目录内附 `snapshot_manifest.json`，记录每个文件的 SHA-256；恢复后可先比对再冻结。生成脚本与修复脚本只作为历史构建记录，不是冻结版复现入口。


