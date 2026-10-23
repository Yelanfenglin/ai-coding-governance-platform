# 工程交付评测

冻结任务输入和判据，使用同一个 `evaluation.runner` 入口。运行缓存、会话、原始输出和
交付文件单独保存，修改判据后重算评分不需要重调模型。

| 场景 | case | 初始工程 |
|---|---|---|
| 从零增加 USART0，保持 PWM/TC1 | `usart_from_scratch.json` | `pwm_before_usart` |
| PWM 40% 与 USART 57600，保留 ADC | `multi_module_change.json` | `pwm_usart_adc` |
| 只读核对当前配置 | `readonly_context.json` | `pwm_usart` |
| 根据上轮指代把占空比改成 40% | `referenced_pwm_change.json` | `pwm_usart` |
| 撤回 40%，最终改为 25% | `corrected_pwm_requirement.json` | `pwm_usart_40` |
| 新会话仅依靠项目记忆回答最终约定 | `fresh_session_memory.json` | `pwm_usart_25` |

这些案例迁自 2026-10-22 的 development 实验，种子保留原字节。厂商源文件沿用现有
fixture 的使用范围；生成源码用于回归。不是新的盲测集，也不是每次新增模型效果。
case 中的 question 是已执行轮次的需求记录；重新做交互评测时，先读本轮实际答复，
再决定下一轮追问或纠正，并保存选择依据，不逐字重放一套预写对话。

从 case 的 `seed` 复制到独立工作目录；验收配置和独立检查器由宿主指定，不由本轮
生成源码决定预期结果。`firmware.py` 复用配置、实际 C 行为、旧业务和工程选项检查；
目标占空比、波特率来自固定任务。`multi_module.py` 在 ADC 检查上核对跨模块改参结果。
需要可用的主机 GCC，可通过 `CC` 指定。

开发中运行：

```text
agent-evaluate check --case <case.json> --workspace <project> --out <checks>
```

交付后评分：

```text
agent-evaluate score --case <case.json> --delivery <delivery.json> --workspace <project> --out <score>
agent-evaluate batch --runs <runs.jsonl> --out <scores>
```

批次每行声明 `case/delivery/workspace`，相对路径以 JSONL 所在目录为基准。
只有确认无人工修复时才设置 `human_interventions=0`。保留原始失败、缺失输入和中间
错误；不得删掉失败运行再计算通过率。

多轮交付共用工程时，每轮保存与其 `delivery.after` 完全一致的 checkpoint。重放历史
轮次必须使用对应 checkpoint，不能拿最后一轮的工程给前一轮评分。评分仍核对 frozen
seed、交付版本、运行结果和验证过程未修改源码。

`fresh_session_memory` 仅用于 `score/batch`：在检查工程未修改之外，还核对 response
确实是新会话、没有文件/历史工具调用、召回过记忆，且结构化答案符合该案例最终约定。
读取历史消息、再查询工程得到正确数字均不符合这项任务。它不评估长期记忆的全部能力。

结果分别报告任务族、运行次数、人工介入、错误与自主修复、耗时、模型用量和压缩次数。
相邻对话轮次算一个交互任务族，不当作多个独立成功任务。主机模拟通过、厂商编译通过、
硬件验证通过需要各自证据。
