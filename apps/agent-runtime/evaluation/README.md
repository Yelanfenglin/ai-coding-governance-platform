# 手动 Agent 任务评估

`runner.py` 对已记录的交付做独立验收，不调用模型。CLI 的 `--delivery-dir` 收集输入、
源文件版本、实际工具与命令、答案和终止状态；运行 completed 和任务通过是两个字段。

`cases/` 是 development 案例，分别来自已有 PWM 参数修改和非模板 USART 接入验收。
`fixtures/pwm_before_usart` 保留当时的初始工程；`fixtures/pwm_usart` 是接受纠正后的工程，
作为后续 PWM 修改的固定起点。保留原文件字节与工具脚本，不能为 lint 改写种子。
这些文件不是新模型运行的结果，也不是厂商 SDK 或板卡验证证据。

执行步骤：复制 case 的 seed 到独立工作目录；使用 case.question 作为任务，不预写工具
顺序；用 CLI 记录交付；运行 `python -m evaluation.runner score --help` 查看评分入口。
批次清单每行给出 `case/delivery/workspace` 和可选 `human_interventions`；路径相对清单。
重复实验可以保留同一 case id，不能省略失败或输入缺失的 run。

判分核对冻结起点、最终文件版本、明确保护的源码、清单与独立主机 C 行为。C harness
沿用原独立验收代码，通过 `CC` 指定主机 GCC；编译输出写到评分目录，验证器不改工程。
未知的人工介入数为 null。任何真实多轮交互应保留实际每轮记录，并按上一答决定追问。
本入口不规定固定台词；跨会话记忆、压缩和取消的机制回归仍复用 `tests/` 中的既有测试。

真实单 Agent/子 Agent 对照必须保留同一初始工程、模型及 thinking/effort、工具、预算和
要求，分别记录任务通过、原功能保持、人工纠正、耗时和总 usage。接线测试和主机验证
不能代替这个效果实验。
