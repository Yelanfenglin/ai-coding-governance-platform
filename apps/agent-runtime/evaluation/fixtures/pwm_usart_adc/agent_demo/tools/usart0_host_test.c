/* Host-side logic test for the USART0 module source (NOT vendor compiled).
 *
 * Standalone: this single file is the whole program. It contains a small
 * stand-in M9P6362 register model plus a verbatim copy ("mirror") of the two
 * functions under test from agent_demo/modules/demo_usart0.c, and
 * `python agent_demo/tools/usart0_calc.py --host-test` first asserts that the
 * mirror region below is byte-identical to the real module source, so the test
 * cannot drift from the shipped code.
 *
 * The target build never sees this file: demo_platform.h keeps its original
 * `#include "zc.h"` path and no host stubs leak into target code.
 *
 * Build & run from the project root:
 *   cc -Wall -Wextra -o usart0_host_test agent_demo/tools/usart0_host_test.c
 *   ./usart0_host_test
 * (or: python agent_demo/tools/usart0_calc.py --host-test, which also builds
 *  and runs it when a host C compiler is available)
 *
 * The register model emulates the spec 10.8.3 p.80 behaviour that matters here:
 * FR0ER / RX0REG describe the CURRENT (oldest unread) byte, and reading RX0REG
 * retires that byte so the buffers shift up to the NEXT one.
 */
#include <stdio.h>

/* ---- stand-in register file ---------------------------------------------- */
static unsigned char TX0CR_r, RX0CR_r, TX0REG_r;
static unsigned char RX0IE_r, RX0IP_r, TX0IF_r;
static unsigned char BRGD0H_r, BRGD0L_r;
static unsigned char ANSA0_r, ANSA1_r, OEA0_r, OEA1_r, PUA1_r;
static unsigned char MPTX0_r, MPRX0_r;

#define TX0CR (TX0CR_r)
#define RX0CR (RX0CR_r)
#define TX0REG (TX0REG_r)
#define RX0IE (RX0IE_r)
#define RX0IP (RX0IP_r)
#define TX0IF (TX0IF_r)
#define BRGD0H (BRGD0H_r)
#define BRGD0L (BRGD0L_r)
#define ANSA0 (ANSA0_r)
#define ANSA1 (ANSA1_r)
#define OEA0 (OEA0_r)
#define OEA1 (OEA1_r)
#define PUA1 (PUA1_r)
#define MPTX0 (MPTX0_r)
#define MPRX0 (MPRX0_r)

/* ---- hardware receive buffer model (spec 10.8.3 p.80) -------------------- */
#define HW_RX_DEPTH 3		/* 3 unread bytes -> RxOVF, 3rd dropped */

static unsigned char hw_data[HW_RX_DEPTH];	/* RX0REG per byte */
static unsigned char hw_frer[HW_RX_DEPTH];	/* FRxER per byte */
static int hw_count;				/* bytes waiting */
static unsigned char RX0IF_r;			/* set on every received byte */
static unsigned char RXOVF_r;			/* set when a 3rd byte arrives unread */

static void hw_reset(void)
{
	hw_count = 0;
	RX0IF_r = 0;
	RXOVF_r = 0;
}

static void hw_receive(unsigned char data, unsigned char frer)
{
	if (hw_count >= HW_RX_DEPTH) {
		RXOVF_r = 1;	/* spec: 舍弃第三个接收数据 */
		return;
	}
	hw_data[hw_count] = data;
	hw_frer[hw_count] = frer;
	hw_count++;
	RX0IF_r = 1;
}

/* FR0ER / RX0REG are reads: they describe the CURRENT byte; reading RX0REG
 * retires it and shifts the rest up. */
#define FR0ER (hw_count > 0 ? hw_frer[0] : 0)
#define RX0REG (hw_rx_read())

static unsigned char hw_rx_read(void)
{
	unsigned char data;

	if (hw_count == 0) {
		RX0IF_r = 0;
		RXOVF_r = 0;
		return 0;
	}
	data = hw_data[0];
	hw_count--;
	hw_data[0] = hw_data[1];
	hw_frer[0] = hw_frer[1];
	hw_data[1] = hw_data[2];
	hw_frer[1] = hw_frer[2];
	if (hw_count == 0) {
		RX0IF_r = 0;	/* 完全读取 RX0REG 后 RXIF 自动清零 */
	}
	RXOVF_r = 0;		/* 读缓冲区后 RxOVF 清零 */
	return data;
}

/* ---- queue definitions, identical to demo_usart0.h ---------------------- */
#define DEMO_USART0_RXQ_SIZE 16u
#define DEMO_USART0_TXQ_SIZE 16u

static volatile unsigned char rx_queue[DEMO_USART0_RXQ_SIZE];
static volatile unsigned char rx_head, rx_tail;
static volatile unsigned char tx_queue[DEMO_USART0_TXQ_SIZE];
static volatile unsigned char tx_head, tx_tail;

/* ==== BEGIN MIRROR: demo_usart0.c :: Demo_Usart0_RxHandler ============== */
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
/* ==== END MIRROR: Demo_Usart0_RxHandler ================================= */

/* ==== BEGIN MIRROR: demo_usart0.c :: Demo_Usart0_Poll =================== */
/* 主循环上下文：发送寄存器空（TX0IF，§5.6 p.43）且队列有数据时推一个字节。 */
void Demo_Usart0_Poll(void)
{
	if (tx_head != tx_tail && TX0IF) {
		TX0REG = tx_queue[tx_tail];
		tx_tail = (unsigned char)((tx_tail + 1u) & (DEMO_USART0_TXQ_SIZE - 1u));
	}
}
/* ==== END MIRROR: Demo_Usart0_Poll ====================================== */

/* ==== BEGIN MIRROR: demo_usart0.c :: Demo_Usart0_WriteByte ============== */
/* 业务调用：入队成功返回1；队列满返回0（丢弃该字节，不阻塞、不死等）。 */
unsigned char Demo_Usart0_WriteByte(unsigned char data)
{
	unsigned char next;

	next = (unsigned char)((tx_head + 1u) & (DEMO_USART0_TXQ_SIZE - 1u));
	if (next == tx_tail) {
		Demo_Usart0_Poll();	/* 阻塞前先推进一次，给队列留出空位 */
		next = (unsigned char)((tx_head + 1u) & (DEMO_USART0_TXQ_SIZE - 1u));
		if (next == tx_tail) {
			return 0;
		}
	}
	tx_queue[tx_head] = data;
	tx_head = next;
	Demo_Usart0_Poll();
	return 1;
}
/* ==== END MIRROR: Demo_Usart0_WriteByte ================================= */

/* ==== BEGIN MIRROR: demo_usart0.c :: Demo_Usart0_ReadByte =============== */
/* 业务调用：出队成功返回1并写出数据；无数据返回0。 */
unsigned char Demo_Usart0_ReadByte(unsigned char *data)
{
	if (rx_head == rx_tail) {
		return 0;
	}
	*data = rx_queue[rx_tail];
	rx_tail = (unsigned char)((rx_tail + 1u) & (DEMO_USART0_RXQ_SIZE - 1u));
	return 1;
}
/* ==== END MIRROR: Demo_Usart0_ReadByte ================================== */

static int failures;

static void check(const char *label, int good)
{
	printf("  [%s] %s\n", good ? "PASS" : "FAIL", label);
	if (!good) {
		failures++;
	}
}

int main(void)
{
	unsigned char b;
	int i;
	int got;

	printf("USART0 module host logic test (spec-model registers, no vendor compiler)\n");

	/* RX: single valid frame */
	rx_head = rx_tail = 0;
	hw_reset();
	hw_receive(0x55, 0);
	Demo_Usart0_RxHandler();
	check("valid byte queued", Demo_Usart0_ReadByte(&b) == 1 && b == 0x55);
	check("RX0IF cleared after reading RX0REG", RX0IF_r == 0);

	/* RX: framing error on the CURRENT byte must be dropped */
	hw_reset();
	hw_receive(0x99, 1);
	Demo_Usart0_RxHandler();
	check("framing-error byte dropped", Demo_Usart0_ReadByte(&b) == 0);
	check("bad byte consumed from hardware buffer", hw_count == 0);

	/* RX: THE ORDER TEST -- 2 bytes waiting, byte0 bad, byte1 good.
	 * Reading RX0REG before sampling FR0ER would retire the bad byte and read
	 * GOOD byte1's status, so 0xB1 would be taken as valid and 0xB1's own
	 * status would be lost. With STEP4 -> STEP6 the bad byte is dropped and
	 * only 0xB2 (the second, good byte) is queued. */
	hw_reset();
	hw_receive(0xB1, 1);	/* bad frame, no stop bit */
	hw_receive(0xB2, 0);	/* good frame */
	Demo_Usart0_RxHandler();
	got = 0;
	if (Demo_Usart0_ReadByte(&b)) {
		got++;
		check("first queued byte is the good 0xB2, not the bad 0xB1", b == 0xB2);
	} else {
		check("first queued byte is the good 0xB2, not the bad 0xB1", 0);
	}
	check("only the good byte was queued", got == 1 && Demo_Usart0_ReadByte(&b) == 0);
	check("hardware buffer drained (1 byte left = 0xB2 -> dropped? no: kept)", hw_count == 1);
	Demo_Usart0_RxHandler();	/* RX0IF still set by 0xB2 */
	check("second interrupt picks up 0xB2", Demo_Usart0_ReadByte(&b) == 1 && b == 0xB2);

	/* RX: more bytes than the software queue -> oldest kept, no overflow */
	rx_head = rx_tail = 0;
	FR0ER_OK:
	for (i = 0; i < (int)DEMO_USART0_RXQ_SIZE + 5; i++) {
		hw_reset();
		hw_receive((unsigned char)i, 0);
		Demo_Usart0_RxHandler();
	}
	got = 0;
	while (Demo_Usart0_ReadByte(&b)) {
		got++;
		if (b != (unsigned char)(got - 1)) {
			check("RX order preserved", 0);
			break;
		}
	}
	check("RX queue holds exactly RXQ_SIZE-1 bytes in order",
	      got == (int)DEMO_USART0_RXQ_SIZE - 1);

	/* TX: queued only when the send register is empty (TX0IF) */
	tx_head = tx_tail = 0;
	TX0IF_r = 0;
	check("WriteByte queues while TX busy", Demo_Usart0_WriteByte('A') == 1);
	check("nothing pushed while TX0IF == 0", TX0REG_r == 0);
	TX0IF_r = 1;
	Demo_Usart0_Poll();
	check("Poll pushes queued byte when TX0IF == 1", TX0REG_r == 'A' && tx_head == tx_tail);

	/* TX: full queue reports 0 instead of blocking or overwriting */
	TX0IF_r = 0;
	for (i = 0; i < (int)DEMO_USART0_TXQ_SIZE - 1; i++) {
		Demo_Usart0_WriteByte((unsigned char)('a' + i));
	}
	check("WriteByte returns 0 when TX queue full", Demo_Usart0_WriteByte('z') == 0);
	TX0IF_r = 1;
	for (i = 0; i < (int)DEMO_USART0_TXQ_SIZE - 1; i++) {
		Demo_Usart0_Poll();
	}
	check("TX queue drains in order (head == tail)", tx_head == tx_tail);

	printf("RESULT: %s\n", failures ? "FAIL" : "PASS");
	return failures ? 1 : 0;
}
