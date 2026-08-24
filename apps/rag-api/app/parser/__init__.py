"""Deterministic PDF + PP-DocLayoutV3 reconstruction.

The parser keeps four facts authoritative:

  1. NativeLine 必须正确        —— 保留 PDF 原生 block/line/span，绝不用 y 聚类拼 line
  2. Anchor binding 必须正确    —— PP proposal 即 anchor，native line 按最大 overlap 绑定
  3. Block order 必须正确       —— text 用 PP order，visual/table 按 y 插入
  4. Ruled table topology 正确  —— 真实 intersections + separator + union-find

补全策略保持确定性：真实 rule 连通组件补寄存器位表、缺失外框自适应、
高置信双栏 order 修复、页边噪声兜底、章节与跨页表 metadata。

It deliberately does not guess VLM semantics, force borderless tables, or
replace the established page reading order with a global y-sort.
"""
