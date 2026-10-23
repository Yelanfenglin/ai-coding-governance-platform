# M9P6362 模块示例工程

规格书 V1.16，官方例程 V1.07。
系统配置：HIRC 32MHz / 8T；所有模块共享此配置。

## 模块与资源

| 模块 | 参数 | 实际配置 | 独占资源 |
|---|---|---|---|
| PWM0 输出 | {"frequency_hz": 2000, "duty_percent": 30, "channels": [0, 2]} | {"frequency_hz": 2000.0, "duty_percent": 30.4, "period_register": 124, "frequency_error_percent": 0.0, "duty_error_points": 0.3999999999999986} | TC0, PWM0, IO:C0, IO:C2 |
| TC1 定时事件 | {"interval_us": 10000, "toggle_pin": "C1"} | {"interval_us": 10000, "period_register": 9999, "interval_error_us": 0} | TC1, IO:C1 |

已有工程保留：无。

频率/时间值按官方固定分频换算，实际配置栏显示寄存器量化后的结果。
IO:A0、IO:C0等为逻辑端口名；请按所选封装核对物理引脚。

## 单独运行

在厂商 ZCW-ICE 中打开 demo.prj，选择 M9P6362。工程依赖厂商工具链的 zc.h，
该SDK头文件不在官方示例包中，不提供伪造替代文件。源码使用UTF-8编码。
main.C 演示一次系统初始化、模块初始化、主循环与唯一的中断入口。

## 接入已有工程

1. 将 modules 下的 .c/.h、demo_app.c/.h 和必要头文件加入工程。
2. 已有系统初始化时保留原初始化；核对32MHz/8T和资源，
   不重复调用 Demo_PlatformInit。
3. 外设初始化处调用 Demo_ModulesInit，主循环调用 Demo_Poll。
   全部中断挂接准备完成后再调用 Demo_EnableInterrupts，或沿用已有全局中断控制。
4. 在已有高/低优先级中断函数中分别调用
   Demo_InterruptHigh / Demo_InterruptLow。
   保留原有中断入口，不把 main.C 中的入口再复制一份。
5. 修改配置请在生成入口更新参数；流程会保留并拒绝覆盖已有人工修改。

PWM接口：Demo_Pwm0_Start / Stop；
SetDuty接收计数ticks，不是百分比。

定时接口：Demo_Timer1_TakeTick 返回是否有待处理事件。
配置GPIO指示时由Poll消费该标志；none时由业务代码消费。标志会合并多次中断，
不作为精确的累计计数器。

## 检查结果

已检查入口/函数唯一性、调用接口、文件依赖及资源占用。
厂商编译：未执行。实机运行：未执行。

官方原文与改动依据见 SOURCES.md；项目状态保存在 demo-project.json。
