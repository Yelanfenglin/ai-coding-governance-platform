# parser

PDF 原生结构 + PP-DocLayoutV3 anchor 的确定性规格书重建器。目标是产出适合 RAG 的可追溯结构，而不是复刻 PDF 排版。

## 不变量

1. `NativeLine` 保留 PDF 原生 block/line/span，不按 y 坐标重新拼接左右栏。
2. 每条非空原生行只绑定一个 anchor；`orphan_text_lines` 与 `duplicate_owned_lines` 必须为 0。
3. TABLE/VISUAL/NOISE 是排他 owner，图表文字不会重复进入正文。
4. 表格先以高置信 rule 冻结基线分类；只有失败的 PP table anchor 在增强 separator 拓扑仍通过重复行列校验时才升级，图片/图表 anchor 不参与该升级。
5. rowspan/colspan 默认保守：先恢复细矩形边线并合并共线片段；只有至少三个原子行/列形成高密度记录锚点，且邻接 separator 节奏多数支持时才兜底拆分。两行歧义、多行表头和纵排分组标签保留。
6. 漏表补提要求同时存在二维重复结构和邻近语义上下文（如前置标题、正文或另一张表）；上下文只增强几何证据，不按标题词义套模板。
7. snapshot bbox 由原生图元、同带标签和相邻 anchor 共同约束；并排图以中线隔离，已识别正文/图题不被截图吞入。
8. 漏图补提要求非填充向量图元形成有二维尺寸的连通组件，并有足够局部文字标签；不得按图题或芯片字段词表判断。
9. 编号标题按 `1 -> H2`、`1.1 -> H3`、`1.1.1 -> H4` 建树；布局标签、相对字号、父子层级、同级编号序列和邻近子标题至少两项相互印证。`注：`、`说明：`、`备注：`、`注意：`、`例：` 和枚举值不进入章节树。
10. 段落重排只发生在同一个 text block 内；新 bullet、标题和句末边界不会被拼接。
11. 双栏检测与阅读顺序逻辑保持独立，本轮后处理不修改 layout block 或列顺序。
12. 恢复后的编号标题形成 `section_id/section_path`；跨页表格只在章节标识、列数和页边位置一致时建立续页链接，标题只作结构上下文，不按标题词义套表格模板。

## 产物

```text
<out>/
├── json/pNNN.json       单页结构、章节路径、table cells、snapshot image_ref
├── md/full.md           便携 Markdown；真实合并单元格用 HTML table
├── snapshots/*.png      与 Markdown/JSON 引用严格一致
├── pages.json           全文页面 IR
└── summary.json         数量、repair 计数与质量不变量
```

结构化表同时提供二维 `rows` 与保真 `cells`；后者包含 `row`、`col`、`rowspan`、`colspan`、`text` 和 `bbox`。页面边界使用 `<!-- pdf-page: N -->` 注释，不作为 Markdown 标题污染章节树。

## CLI

```powershell
python -m app.parser.parse `
  --pdf "C:\path\PRODUCT.pdf" `
  --layout "artifacts\paddle_doclayout_v3\PRODUCT_doclayout_v3_regions.json" `
  --out "storage\parsed\PRODUCT" `
  --dpi 160
```

可用 `--pages 6,15,71` 做局部回归，或用 `--no-native-table-recovery` 关闭 PP 漏表补提。默认不容忍 layout 缺页；仅诊断时使用 `--allow-missing-layout`。

## 测试

普通单元测试不依赖私有 PDF。设置以下环境变量可运行高风险真实页面回归：

```powershell
$env:RAG_PARSER_FIXTURE_PDF="C:\path\M8P6272_SPEC_V1.10CN.pdf"
$env:RAG_PARSER_FIXTURE_LAYOUT="artifacts\paddle_doclayout_v3\M8P6272_doclayout_v3_regions.json"
$env:RAG_PARSER_M9_FIXTURE_PDF="C:\path\M9F6820_SPEC_v1.07CN.pdf"
$env:RAG_PARSER_M9_FIXTURE_LAYOUT="artifacts\paddle_doclayout_v3\M9F6820_doclayout_v3_regions.json"
$env:RAG_PARSER_M8P5170_FIXTURE_PDF="C:\path\M8P5170_SPEC_V1.06CN.pdf"
$env:RAG_PARSER_M8P5170_FIXTURE_LAYOUT="artifacts\paddle_doclayout_v3\M8P5170_doclayout_v3_regions.json"
python -m pytest -q
```
