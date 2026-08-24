# 规格书 PDF 解析管线：流程与处理技巧

> 本文说明 `apps/rag-api` 当前解析器的完整处理流程、各阶段职责和关键技术点。
> 主要对象是中文 MCU 数据手册这类**原生文字层产品规格书**；目标是产出**可追溯、机器可读**
> 的结构（Markdown / JSON），供后续 RAG 的 chunking 与检索消费，而不是复刻 PDF 排版。

## 1. 总览

```text
PDF
 └─ PP-DocLayoutV3 版面 anchor（bbox / label / order，离线批处理，断点续跑）
 └─ PyMuPDF 原生层（block / line / span / vector / image；span 保留字符级 bbox）
      ├─ 唯一 ownership：每条非空原生行绑定一个 anchor，漏检行生成 auto text anchor
      ├─ 表格重建：真实 separator 几何优先，文字仅作保守兜底
      ├─ 表格快照 / 视觉补提：模型定位 + 原生图元 / 同带标签扩框 + 相邻 anchor 分界
      ├─ 字符级跨格分配：一个 span 跨多个 cell origin 时按字符坐标拆格
      ├─ 跨页表格拼接：同一逻辑表的分页片段物理合并（pdf-pages + source_pages）
      ├─ 目录区域重建：停止段落重排，按视觉基线一行一项
      ├─ 编号标题恢复：布局标签 / 字号 / 层级序列 / 同级编号 / 邻近子标题多证据打分建树
      └─ 块内语义换行：合并视觉折行，bullets / 标题 / 句末边界保行
 └─ 渲染：Markdown 表、HTML 表（含 rowspan/colspan）、图片引用、pdf-page / pdf-pages 注释
 └─ 产物：json/pNNN.json（单页结构）· pages.json（全文）· md/full.md · snapshots/ · summary.json
 └─ 审计：audit_reconstruction.py（ownership / 表格边界 / 语义行来源 / 图片引用 / 页序 / 标题树）
```

不同内容使用各自的最基本结构，互不混淆：

| 内容 | 重建方式 |
|---|---|
| 普通正文 | paragraph reconstruction（块内合并视觉换行） |
| 表格 | cell / grid reconstruction（+ 跨页 stitch + 无边框嵌套子表语义视图） |
| 目录 | visual-line reconstruction（一条视觉行 = 一条目录项） |
| 图片 | placeholder（截图 + annotations） |

## 2. 各阶段说明

### 2.1 版面 anchor（离线）

`scripts/pp_doclayout_all.py` 用 PP-DocLayoutV3 输出每页区域的 `bbox / label / order`，
支持断点续跑与原子 checkpoint。产物是**候选**，不是真值：轻量模型把大量封装图、波形图、
代码块标成 `table`，也会漏检真表格。因此下游所有“真/假”判定都以几何结构为准，
模型 label 只参与 anchor 生成与阅读顺序。

### 2.2 原生提取

PyMuPDF 原生层提取 `block / line / span`（保留字符级 `bbox`）、vector 绘图对象与 image。
原则：

- **不按 y 坐标重新拼接原生行**——原生行是版面真相的基线。
- 每条 `NativeSpan` 额外保留 `characters`（每字符 bbox），供 cell 级分配与目录视觉行重建使用。
- vector 层区分填充矩形（常被用作表格细边框）与轮廓线，供几何重建使用。

### 2.3 唯一 ownership

每条非空原生行寻找“最匹配”的 layout anchor（面积重合率），绑定后：

- TABLE / VISUAL / NOISE 是**排他 owner**：图表、页眉页脚、页码文字不会重复进入正文。
- 没有任何 anchor 命中的行生成 `auto` text anchor（PP 漏检兜底）。
- 不变量：`orphan_text_lines == 0` 且 `duplicate_owned_lines == 0`，审计器强制校验。

### 2.4 表格重建（本管线最复杂的部分）

分层决策，逐层收紧：

1. **分类冻结**：先用旧版高置信 rule 边线做出 table / snapshot 的基线分类。
   轻量模型的历史分值可作为参考，但最终判定来自几何（重复二维记录）。
2. **失败升级**：只有基线失败的 PP table anchor，在“增强 separator 拓扑仍通过重复行列校验”
   时才从 snapshot 升级为结构化表；图片/图表 anchor 不参与该升级。
3. **边线恢复**：细填充矩形（0.5–2pt）恢复为 rule；同一边界上的分段 rule 先做区间并集，
   再聚类成 row / col 边界。
4. **原子格与 merge**：以真实横竖线铺出 atomic cell；缺失 separator 的相邻格形成候选
   rowspan / colspan；**宁可不合并，也不要错合并**：
   - 候选 merge 内若多个原子行/列各自有文字，按文字起点重新分段；
   - 文字拆分的兜底必须同时满足：≥3 行/列的高密度记录锚点 + 邻接 separator 节奏多数支持；
   - 两行歧义单元格、多行表头、纵排分组标签保持原跨度。
5. **漏表补提**：全页几何扫描发现的漏表，需要“重复二维结构 + 邻近语义上下文”
   （前置标题 / 正文 / 邻近表格）同时成立；上下文只增强几何证据，不按标题词义套模板。
6. **无边框嵌套子表**：外层 cell 内没有横线、但各列物理行数相同且跨列基线对齐时，
   生成规范化 `semantic_rows / aligned_subrows`。原始 outer topology 保留为保真视图；
   单个多行 cell、表头换行、普通折行不升级。**该判定在解析层一次性完成，不交给 chunker 二次猜测。**

### 2.5 图片与表格快照

- **不要照搬模型 bbox**：模型给的框常只包住芯片主体，左右引脚标签被裁。
- 扩框证据：原生 PDF 图元、同一水平带的近邻短标签、相邻视觉 anchor 的中点（并排图互相隔离）。
- 边界约束：已被 layout 识别为正文 / 图题的 anchor 构成边界，避免正文被截图吞入。
- **纯矢量漏图**：非填充矢量图元形成有二维尺寸的连通组件 + 足够局部文字标签时才补提；
  文档名、页码、图题词表一律不进生产规则。
- 剩余无法可靠结构化的区域保留截图 + `annotations`（图内原生文字），
  Markdown 引用与 JSON `image_ref` 严格一致。

### 2.6 字符级跨格分配

PDF 生成器偶尔把多个视觉单元格编码进**一个 text span**：
整 span 中心归属会把整列 label 塌进一个 cell。解决：

- 保留 span 的字符级 bbox；字符按坐标落入哪个 atomic cell，就分配到那个格。
- 只有当字符分组对应**多个不同 cell origin** 时才拆格；
  真实 rowspan/colspan 覆盖的原子格共享同一 origin，因此完整保留原 span。

### 2.7 跨页表格拼接

相邻页同一张表的物理分页片段合并为一张逻辑表。**四个条件缺一不可**：

1. 上一页最后一个正文 block 是 `table`；
2. **下一页第一个正文 block 就是 `table`**（`is_first_body_block`；
   这一条杀死相邻页大量相同 `Bit7..Bit0` 表头、但被真章节标题引入的寄存器表）；
3. 表头取**最长相同行前缀**（去空白归一化，`单位` == `单 位`），下一页去掉重复表头；
4. 列数一致 + 水平 span 一致（bbox 仅作保险，不是主证据）。

- 支持任意长链（11→12→13）。
- 合并后输出 `<!-- pdf-pages: N-M -->` 注释，且在 `TableBlock.source_pages` 记录来源页。
- **明确不做**：rowspan/单元格被页切断的拼接、表格截图（snapshot）的跨页合并。
- 被整页吸收（无剩余内容）的页面不再产生独立 `<!-- pdf-page: N -->` 标记。

### 2.8 目录区域重建

目录没有“段落”概念，普通 reflow 会把根级目录项（`1 产品简述`，非 bullet、无句末标点）
拼进上一行。解决：

- **探测**：页面含 `目录` 标题，或 ≥5 条“点线 `...` + 末尾页码”视觉行 → TOC 模式。
- **重建**：停止段落重排；一条视觉行 = 一条目录项；
  **末尾页码 = 硬闭合**（永不与下一行拼接）；
  仅当前一行无页码、下一行带页码时保守拼接长标题折行。
- 点线与页码阶段一保留（忠实解析优先），不参与章节树。

### 2.9 编号标题恢复

视觉标题识别不错，但“是不是标题、几级标题”需要全文语境。单条正则容易把
`0 = 关闭`、`8 位模式`、日期误升标题。解决：把五个证据打分求和：

- 布局标签（`paragraph_title` 等）与相对字号；
- 与已接受的章节层级（父子关系、同级编号连续、根编号序列）；
- 邻近子标题（后代存在）。

`1 → H2`、`1.1 → H3`、`1.1.1 → H4`。证据不足一律驳回；
负向前缀（`注：` / `说明：` / `备注：` / `注意：` / `例：`）与枚举值不进章节树。

### 2.10 块内语义换行

仅在同一 text block 内合并 PDF 视觉换行：新 bullet、编号标题、句末标点（`。！？；!?;：:`）
保行为界；中文直接连接，英文按前后字符决定是否补空格。

### 2.11 渲染与产物

- 无 rowspan/colspan 的表 → Markdown 表；含 merge 的表 → HTML `<table>`（保留 `rowspan/colspan`）。
- 语义子表视图（`semantic_rows`）用于 Markdown；`cells`（保真）在 JSON 中始终提供。
- 页面边界用 `<!-- pdf-page: N -->` 注释，不进 Markdown 标题树；
  跨页表用 `<!-- pdf-pages: N-M -->`。
- `summary.json` 提供计数与质量不变量（orphan / duplicate / missing snapshot / continuation）。

## 3. 泛化约束（最重要的一条）

生产代码**只依据几何、文本形态和邻接/上下文一致性**：

- 禁止出现文档名、页码、章节标题词义、引脚名（`IOA`/`PWM`/`TK`…）或任何“芯片字段词表”。
- 这些内容只允许出现在**回归断言**里（固定样本的验收），绝不进入判定逻辑。
- 层级按“最接近真实证据”设计：几何 > 上下文 > 文字。
