# Agent 与 RAG 验收摘要（2026-10-23）

已验证的工作闭环是：读取工程和规格书证据、计算配置、修改本地文件、执行宿主提供的
独立检查、根据失败输出自行修复，并在后续追问中读取当前工程和历史上下文。
结论适用于本页的任务与检查范围，不等同于生产稳定性或硬件验证。

## 真实模型与真实 RAG

使用 DeepSeek Flash、thinking enabled、reasoning_effort high、1,000,000 tokens 窗口。
本批未使用 Pro、脚本模型或 RAG 替身。没有人工修改交付代码，也没有在运行中修改
runtime 或独立检查器。

从冻结的 PWM0＋TC1 工程开始，为 M9P6362 V1.16 增加 USART0：IOA0/IOA1、115200、
8N1、非阻塞软件队列、高优先级接收、坏帧丢弃与轮询发送，同时保留原功能。
Agent 实际调用 RAG 核对停止位、波特率公式、中断优先级与标志位语义，再生成代码。
开发检查先失败两次：文件登记不完整、缺少函数声明；模型读取输出后自行修复。
最终版本通过独立主机行为、文件保护与清单检查。

首轮完成后，先阅读真实答复，再追问“16 字节软件队列是否保证连续接收 16 字节”。
第二轮续接历史，重新查手册并读源码，正确区分硬件溢出与留一槽判满的 15 字节容量，
保持工程不变。答案中的“不是 FIFO 阵列”缺少足够证据，保留为本次措辞缺陷，
不计为已证实的结论。

| 本批测量项 | 结果 |
|---|---:|
| 工程任务族／后续追问 | 1／1 |
| 真实模型请求 | 62 |
| search_spec／read_context | 10／1 |
| 模型总 tokens | 5,010,185 |
| 输入缓存命中／未命中 tokens | 4,742,400／209,138 |
| 输出 tokens | 58,647 |
| 最大单次输入 tokens | 135,568 |
| 自动压缩次数 | 0 |

用量来自 provider 返回，不含 RAG 查询 embedding 的费用，也不是账单金额。
首轮存在自写校验脚本的正则误判与重复检查，工作效率仍有改进空间。
两个 turn 属于同一任务族，不据此计算泛化成功率；本批也没有触发真实长任务压缩。

## 自动验证与交付入口

默认测试覆盖流式 transport、工具调用配对、并行与取消、压缩输入和重放、记忆隔离、
会话恢复、RAG 工具、子 Agent 和交付评分。固定脚本 HTTP 对端另验证真实 CLI 进程：
失败反馈后修复可以交付；直接声称完成、或检查后再次改坏文件，宿主均拒绝交付。
已有三个任务族的六个冻结运行可离线重评，不为重新计算评分调用模型。

项目环境下运行，主机固件测试另需 C 编译器，必要时用 `CC` 指定：

```powershell
$env:PYTHONNOUSERSITE='1'
New-Item -ItemType Directory -Force output | Out-Null
.venv/Scripts/python.exe -m pytest -q --basetemp output/pytest-publish
.venv/Scripts/python.exe -m ruff check app tests evaluation
.venv/Scripts/python.exe -m evaluation.runner --help
```

检查器、case、测试种子与协议随源码提交。复跑真实任务需要自行配置模型、RAG 服务、
官方例程和目标工程；检索库与付费运行缓存不是源码包的一部分。

## 原始记录与边界

本批原始记录保存在本地，未入库：`output/rag_agent_live_20261023/` 包含请求、SSE、
会话、交付、各轮冻结版本、引用来源核对与评分；`output/delivery_entry_20261023/`
包含自动测试、CLI 接线和打包验证。RAG 冷启动修复记录位于服务自己的本地
`output/rag_recovery_20261023/`；部署配置与根因说明已提交于其 `infra/milvus/`。

验证层限于主机 C 编译、独立行为判据、文件保护与引用核对；未使用厂商编译器或板卡。
本地命令沿用当前用户权限，没有 OS 沙箱。本批通过不补齐尚未实现的崩溃事务恢复、
服务端任务管理或生产部署能力；具体源码适配边界见
[实现对照](source-alignment-agent-delivery.md) 与 [README](../README.md)。
