# 来源与适配

官方例程：[M9P6362/M9P6362_Demo_V1.07.zip](http://www.masses-chip.com/3090.html)，版本V1.07。
规格书：M9P6362 V1.16。例程版本与规格书版本分别记录。

保留已有工程的系统初始化及全部工程选项；模块不重设时钟或WDT/复位配置。
当前工程选项和OSCM写入已与审查过的32MHz/8T配置对照。
新增模块初始化期间暂时关闭GIEH并恢复原值；新定时模块初始化后开放高优先级中断。

## PWM0 输出（C0，20 kHz / 30 %）
包内源码：`M9P6362_Demo_V1.07/M9P6362_PWM0/PWM0_Work.c`。
规格位置：§7.1 p.55（TC0 8位模式：TC0CL为周期寄存器，TC0CH计入到TC0CL后溢出）、
§7.2 p.56（T0CR：TC0EN/TC0MOD/TC0CKS/TC0CKPS）、§7.3 p.57（T0CR2的TC0CKPO后分频）、
§7.6–7.7 p.58（PWM00CR/PWM00D）、§7.8 p.59（波形示例：00H上升、PWM00D下降）、§6.9.3 p.54（MPPWM0引脚映射）。

当前参数与依据（由 `agent_demo/tools/pwm0_calc.py` 枚举 2048 种组合得出，不是手算）：

| 项 | 值 | 依据 |
| --- | --- | --- |
| 时钟源 | TC0CKS=01 → FOSCH（HIRC 32 MHz，未改动） | §4.1 p.33、§7.2 p.56 |
| 预分频 | TC0CKPS=011 → 1:8 | §7.2 p.56 |
| 计数模式 | TC0MOD=0 → 8位（PWM0使能要求） | §7.1 p.55、§7.6 p.58 |
| 周期寄存器 | TC0CL=199 → 周期200个计数 | §7.1 p.55（周期=TC0CL+1） |
| 占空比寄存器 | PWM00D=60 | §7.7 p.58、§7.8 p.59（占空比=PWM00D/(TC0CL+1)） |
| 实际频率 | 32 MHz/(8×200)=20000.0000 Hz，误差0.0000 % | 同上 |
| 实际占空比 | 60/200=30.0000 %，200个占空比计数步 | 同上 |
| 高/低电平时间 | 15.000 µs / 35.000 µs，周期50.000 µs | 同上 |

寄存器写入：`MPPWM0=0x20`（PWM00→IOC0=C0，§6.9.3 p.54）、`T0CR=0x0B`、`T0CR2=0`、`TC0CL=199`、
`TC0CH=0`、`PWM00CR=0x80`、`PWM00D=60`，最后单独写 `TC0EN=1`。

初始化顺序与官方例程一致：先配置再开启。官方 `PWM0_INIT()` 先 `T0CR=0x0F`（TC0EN=0）、
`TC0CL`、`TC0CH`，再 `TC0EN=1`，随后写 `PWM00CR`/`PWM00D`；本模块同样让周期与占空比写入落在
TC0 启动之前，只把 `TC0EN=1` 单独放在末尾，避免启动后立即改周期/占空比。

`T0CR2=0` 的说明：只用于把后分频保持在复位默认的 1:1（官方例程同样不写 T0CR2）。
§7.3 p.57 只说明后分频"在原来预分频的基础上再分频"；本工程未查到后分频器与 PWM 输出生成之间
相互作用的明文图/文，因此计算脚本不依赖也未假设该语义——频率公式只用到 TC0CKPS 预分频
（§7.2 p.56）和 TC0CL+1（§7.1 p.55）。若将来需要启用后分频，须先补证。

复跑校验：`python agent_demo/tools/pwm0_calc.py --check`（含源码宏/寄存器与"配置→开启"顺序的一致性检查）。

适配：仅初始化所选通道；保留官方C0–C3映射；移除50ms开关/延时；显式复位TC0后分频；
按固定预分频换算周期与占空比并报告实际值。

## USART0 异步串口（TX→A0，RX→A1，115200 8N1）
包内源码：`M9P6362_Demo_V1.07/M9P6362_Usart0/Usart0_Work.c`。
规格位置：§10.2 p.74（TXxCR：TXxEN/TxMCLR/TxSYNC/TxL9/TxSPD[1:0]）、§10.4 p.75（RXxCR：RXxEN/RxL9/RxOVF/FRxER）、
§10.5 p.76（RXxREG 只读，读后清 RXxIF）、§10.6 p.76（BRGDxH：STOPx1 等）、§10.7 p.76（BRGDxL）、
§10.8.1 p.77（波特率公式与 32 MHz 表：115200 → n=4、BRGD=0x44、偏差0.64 %）、
§10.8.3 p.80（异步接收流程 STEP1–STEP6）、§5.5 p.42（INTCR0：RX0IE/TX0IE）、§5.6 p.43（INTF0：RX0IF/TX0IF）、
§5.7 p.44（INTP0：RX0IP，0=高优先级）、§6.9.1/§6.9.2 p.53（MPTX0/MPRX0 全端口映射）。

当前参数与依据（由 `agent_demo/tools/usart0_calc.py` 枚举 4×1024 种组合得出，不是手算）：

| 项 | 值 | 依据 |
| --- | --- | --- |
| 分频 n | TxSPD[1:0]=00 → n=4 | §10.2 p.74、§10.8.1 p.77 |
| 波特率寄存器 | BRGD=0x044（68） | §10.6/§10.7 p.76 |
| 实际波特率 | 32 MHz/((0x44+1)×4)=115942.0290 bps | §10.8.1 p.77 |
| 波特率偏差 | +0.6441 %（≤ 2 %） | 同上（规格表给 0.64 %） |
| 帧格式 | TX0CR=0x80：TX0EN=1、TxSYNC=0 异步、TxL9=0 8位数据 | §10.2 p.74 |
| 停止位 | BRGD0H=0x20：STOP1=1 → 1 个停止位（8N1） | §10.6 p.76 |
| 接收 | RX0CR=0x80：RX0EN=1、RxL9=0 8位、SRxEN=0 异步 | §10.4 p.75 |
| 位时间/帧时间 | 8.6250 µs / 86.2500 µs（8N1 共 10 位） | §10.8.1 p.77 |
| 引脚映射 | MPTX0=0x00 → IOA0；MPRX0=0x01 → IOA1 | §6.9.1/§6.9.2 p.53 |
| 引脚方向 | OEA0=1（输出）、OEA1=0（输入）、PUA1=1（接收上拉）、ANSA0/ANSA1=0（数字口） | §6 端口寄存器、§1.2.6 注(1) p.14 |
| 中断 | RX0IE=1、RX0IP=0（高优先级）；TX0IE 不开，发送用 TX0IF 轮询推进 | §5.5–§5.7 p.42–44、§5.6 p.43 |

时钟口径：公式里的 `FOSC` 按本工程 HIRC 32 MHz（`OSCM=0x00` 未改、工程选项未改）代入，
与 §10.8.1 p.77 的 32 MHz 表一致；`OSCM` 写入保持不变，未切换时钟源。

`STOP1` 必须显式写 1：BRGDxH 复位值 bit5=0 表示 2 个停止位（§10.6 p.76）。官方例程
`Usart0_Work.c` 只用 `BRGD0H |= (…)` 合并高两位、并未置 STOP1，直接照抄会得到 2 个停止位而不是 8N1，
因此本模块改为 `BRGD0H = 0x20`（`=` 而非 `|=`），把 STOP1 一并定义。此点官方例程（V1.07 包）
与规格书写法不一致，以规格书为准记录在此，不把例程行为当依据。

接收顺序（spec §10.8.3 p.80 STEP4→STEP6）：先判 `FR0ER`（当前待读字节的帧格式状态），再读 `RX0REG`。
反序会先读走数据、把状态前移到下一帧，缺停止位的坏帧会被当作正确数据入队；本模块按 STEP4→STEP6 实现，
`Demo_Usart0_RxHandler()` 内 `FR0ER` 出现在 `RX0REG` 之前，并且只有 `FR0ER==0` 分支才入队。

适配：引脚由例程的 IOB3/IOB4 改到 IOA0/IOA1（C0 已被 PWM0 占用），映射合法性见 §6.9.1/§6.9.2 p.53；
改为非阻塞队列接口（`Demo_Usart0_WriteByte`/`Demo_Usart0_ReadByte`，返回1=成功0=队列满/空），
移除例程的 `Delay_50ms` 回显；发送由 `Demo_Usart0_Poll()` 在 `Demo_Poll()` 中按 `TX0IF` 推进，不开 TX 中断；
接收在 `Demo_InterruptHigh()` 中处理（`RX0IP=0` 与 TC1 同一高优先级向量，`TC1_Interrupt()` 仍先执行）。

复跑校验：`python agent_demo/tools/usart0_calc.py --check`（含宏/寄存器值、初始化顺序、
STEP4→STEP6 判断次序、`main.C` 主循环 `Demo_Poll()` 与 `int_h()` 的 `Demo_InterruptHigh()` 接入、`.prj` 文件清单）。
主机逻辑验证：`agent_demo/tools/usart0_host_test.c`（在项目根目录用任意主机 C 编译器单独编译，配合
`usart0_calc.py --host-test` 的替代寄存器模型运行；不参与目标工程编译，也不改 `demo_platform.h`）。

## ADC0 单次采样（IOB7/AD15，VDD 参考，12 位右对齐）

包内源码：`M9P6362_Demo_V1.07/M9P6362_ADC/ADC_Work.c`（`ADC_INIT`/`ADC_CONVERTED`/`ADCDATA_AVERAGE`/`ADCSetADVOS`）。
规格位置：§13.2 p.99（ADCON0：b7 ADEN、b6 ADS、b5 ADFM、b4:0 CHS）、§13.3 p.100（ADCON1：b7 AGEN、
b6:4 ADCKS、b3:2 VREMS、b1:0 VHS，ADCKS 111 = Fcpu/128，VREMS 00 = VDD）、
§13.5 p.102（ADH/ADL：ADFM=1 时 ADRES = {ADH[3:0], ADL[7:0]}）、§13.6 p.102（12 位转换 = 16 个 ADC clock）、
§1.3 p.15–17（IOB7 = AD15）、§6.6 p.51（ANSx：1 = 模拟口）、§4.4 p.35（8T → Fcpu = FOSC/8 = 4 MHz）。

| 项 | 实际值 | 依据 |
| --- | --- | --- |
| 通道 | CHS[4:0]=01111 → AD15（IOB7） | §13.2 p.99、§1.3 p.15–17 |
| ADCON0 | `0xAF`：ADEN(b7)=1、ADS(b6)=0、ADFM(b5)=1（右对齐） | §13.2 p.99 |
| ADCON1 | `0x70`：AGEN=0、ADCKS=111（Fcpu/128）、VREMS=00（VDD）、VHS=00 | §13.3 p.100 |
| ADC 时钟 | 31250 Hz（4 MHz/128，Fcpu 口径）；按 FOSC 口径为 250 kHz；均 ≤ 1 MHz | §13.3 p.100、§4.4 p.35 |
| 转换时间 | 16 个 ADC clock = 512 µs（Fcpu 口径）／64 µs（FOSC 口径） | §13.6 p.102 |
| 端口设置 | ANSB7=1、PUB7=0、PDB7=0、OEB7=0，只动 B7 | §6.6 p.51、§13.2 p.99 注 |

ADCON0 位位置必须按原文解码，不能照抄"ADFM 在 bit6"的直觉：§13.2 p.99 明确 b6 是 **ADS**（写 1 立即启动转换）、
b5 才是 **ADFM**。若把 ADFM 写到 bit6（即 `ADCON0=0xCF`），会同时得到 ADS=1（Init 阶段误启动转换）和
ADFM=0（左对齐，结果落在 ADH/ADL 高位），因此本模块写 `0xAF`，并在 `adc0_calc.py` 中按位位置解码复核。
官方 `ADC_INIT()` 的 `ADCON0 = 0xAF` 与此一致；例程的 `ADCON1 = 0x25` 是内部 2 V 参考，本模块按需求改为 VDD 参考（`0x70`），
官方 `while(ADS==1)` 轮询与 10 次去极值求平均在 `ADC_CONVERTED`/`ADCDATA_AVERAGE` 中，本模块不沿用。

接口与状态机（`Demo_Adc0_Start/Poll/Read`）：Init 只写 ADCON1/ADCON0（ADEN=1、ADS=0）与引脚，并做有限稳定延时；
Start 在 IDLE 时才写 `ADS=1` 并返回 1，转换中或结果未取走时返回 0 且不重启；Poll 只在 `ADS` 已被硬件清零时把
`AD`（右对齐 12 位）存入内部结果，不忙等、不自动连续启动；Read 无新结果返回 0 且不改 `*value`，有结果时取走一次后回到 IDLE。
无新结果被覆盖的问题由"结果未取走不接受新 Start"保证。

未验证的边界：本轮只做原始数字采样，未移植依赖真实转换完成的 `ADCSetADVOS` 失调校准（官方用 GND 通道实测迭代），
因此失调/增益/INL/DNL、输入源内阻与采样保持影响、VDD 参考随电源波动的误差均未测量；
无厂商编译器与板卡，仅做了静态/算术与主机替代寄存器验证，**实际精度需板级实测**（可用已知电压源扫点比对
LSB = VDD/4096 的偏差）。转换完成的等待由调用者的轮询节奏决定，最坏情况为一个主循环周期。

复跑校验：`python agent_demo/tools/adc0_calc.py --check`（含寄存器位解码、无忙等、引脚只动 B7、
`Demo_ModulesInit`/`Demo_Poll` 接入、`.prj` 文件表与 `demo-project.json` 实际值一致性）。

## 未改动的部分
系统时钟与工程选项（`M9P6362_TC1.prj` 未改）、TC1 的 1 ms 事件（`TC1_Work.c` 未改）、
`main.C` 中的 `Project_Work()` 业务代码（`business_cycles += 2;` 未改）。
`main.C` 仅新增两处接出：主循环里的 `Demo_Poll();` 和 `int_h()` 里的 `Demo_InterruptHigh();`（`TC1_Interrupt()` 仍在前）。
