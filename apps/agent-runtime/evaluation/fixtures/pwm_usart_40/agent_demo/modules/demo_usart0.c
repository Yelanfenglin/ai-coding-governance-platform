/* Derived from MASSES CHIP M9P6362 Demo V1.07.
 * Original work: (C) COPYRIGHT 2025 Masses-Chip.
 * Official source: http://www.masses-chip.com/3090.html
 * Modular adaptation and changed parameters: see SOURCES.md.
 */
#include "demo_usart0.h"

/* Receive queue filled by the interrupt handler, drained by the user code.
 * Single producer (ISR) / single consumer (main loop) -> the head index is
 * only written by the ISR and the tail index only by the main loop. */
static volatile unsigned char rx_queue[DEMO_USART0_RXQ_SIZE];
static volatile unsigned char rx_head;
static volatile unsigned char rx_tail;

/* Transmit queue written by the user code, drained from Demo_Usart0_Poll(). */
static volatile unsigned char tx_queue[DEMO_USART0_TXQ_SIZE];
static volatile unsigned char tx_head;
static volatile unsigned char tx_tail;

void Demo_Usart0_Init(void)
{
	//1、USART0管脚设置
	//USART0 TX/RX 为全端口映射（spec 6.9.1/6.9.2 p.53）：
	//MPTX0/MPRX0 的 PORTSEL[1:0]=bit[5:4] 选端口组(00=IOA)，PINSEL[2:0]=bit[2:0] 选管脚号
	//MPTX0 = 0x00 -> TX0 映射到 IOA0（逻辑 A0）
	//MPRX0 = 0x01 -> RX0 映射到 IOA1（逻辑 A1）
	//与官方例程不同：例程用 MPTX0=0x20/MPRX0=0x21（C0/C1），本工程 C0 已被 PWM0 占用，故改到 A0/A1
	ANSA0 = 0;
	ANSA1 = 0;
	OEA0 = 1;	//TX 输出
	OEA1 = 0;	//RX 输入
	PUA1 = 1;	//RX 上拉，避免悬空误触发接收中断（例程对 RX 引脚也是这么做的）
	MPTX0 = 0x00;
	MPRX0 = 0x01;
//2、波特率设置：32MHz / ((0x44+1) * 4) = 115942 bps，偏差 +0.64 %（要求 <= 2 %）
	/* TX0CR = 0x80: TX0EN=1, TxSYNC=0 (异步), TxL9=0 (8 位数据), TxSPD=00 (n=4), TxD9=0。
	 * TxMCLR(bit6) 为只读的"发送寄存器空"标志，不写入。 */
	TX0CR = 0x80;
	/* BRGD0H 直接赋值而不是 "|="：bit5 STOP1 复位值为 0（2 位停止位），
	 * 只有显式写 1 才是 1 位停止位，帧格式才符合 8N1。bit[1:0]=BRGD[9:8]=00。 */
	BRGD0H = DEMO_USART0_BRGD0H;
	BRGD0L = DEMO_USART0_BRGD0L;
//3、接收设置
	/* RX0CR = 0x80: RX0EN=1, RxL9=0 (8 位数据), SRxEN=0 (异步接收无需启动同步接收) */
	RX0CR = 0x80;
	/* 清队列，再清接收标志（RX0IF 由硬件在读取 RX0REG 后自动清零，这里先写 0 防止残留），
	 * 最后才使能 RX0IE 并选择优先级，和官方例程顺序一致：先配置，后开中断。 */
	rx_head = 0;
	rx_tail = 0;
	tx_head = 0;
	tx_tail = 0;
	RX0IF = 0;
	RX0IE = 1;
	RX0IP = 0;	//0 = 高优先级，与 TC1 同在 Demo_InterruptHigh() 中处理
}

/* 高优先级中断上下文（由 main.C 的 int_h() 经 Demo_InterruptHigh() 调用）。
 * 顺序按 spec 10.8.3 p.80 的 STEP4/STEP6：先判断 FR0ER（当前待读字节的状态），
 * 再读 RX0REG（读取后才把当前字节移出，并自动清 RX0IF；RXOVF 也在读缓冲区后清零）。
 * 若先读 RX0REG，当前字节的状态（FR0ER/RxD9）已前移到下一个字节，
 * 格式错的那一帧就会被当成正确数据收下。 */
void Demo_Usart0_RxHandler(void)
{
	unsigned char data;
	unsigned char next;

	if (RX0IF) {
		if (FR0ER == 0) {
			next = (unsigned char)((rx_head + 1u) & (DEMO_USART0_RXQ_SIZE - 1u));
			if (next != rx_tail) {
				rx_queue[rx_head] = RX0REG;	/* 读 RX0REG：清 RX0IF/RXOVF */
				rx_head = next;
			} else {
				/* 软件队列满：仍要读走 RX0REG，避免硬件缓冲溢出，丢弃该字节 */
				data = RX0REG;
				(void)data;
			}
		} else {
			/* 帧格式错（缺停止位，见 10.4 p.75 FRxER）：读走丢弃 */
			data = RX0REG;
			(void)data;
		}
	}
}

/* Non-blocking: TX0IF = 1 表示TX0REG发送缓冲为空，可写入下一个字节。
 * TX0IF与TX0CR bit6 TxMCLR（移位寄存器空）是两个状态，本函数只轮询缓冲可写状态。 */
void Demo_Usart0_Poll(void)
{
	if (tx_head != tx_tail && TX0IF) {
		TX0REG = tx_queue[tx_tail];
		tx_tail = (unsigned char)((tx_tail + 1u) & (DEMO_USART0_TXQ_SIZE - 1u));
	}
}

unsigned char Demo_Usart0_WriteByte(unsigned char data)
{
	unsigned char next;

	Demo_Usart0_Poll();	/* 先把已排队的数据推出去 */
	next = (unsigned char)((tx_head + 1u) & (DEMO_USART0_TXQ_SIZE - 1u));
	if (next == tx_tail) {
		return 0;	/* 队列满 */
	}
	tx_queue[tx_head] = data;
	tx_head = next;
	Demo_Usart0_Poll();
	return 1;
}

unsigned char Demo_Usart0_ReadByte(unsigned char *data)
{
	if (rx_tail == rx_head) {
		return 0;	/* 无数据 */
	}
	*data = rx_queue[rx_tail];
	rx_tail = (unsigned char)((rx_tail + 1u) & (DEMO_USART0_RXQ_SIZE - 1u));
	return 1;
}
