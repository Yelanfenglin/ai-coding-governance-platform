/* Derived from MASSES CHIP M9P6362 Demo V1.07.
 * Original work: (C) COPYRIGHT 2025 Masses-Chip.
 * Official source: http://www.masses-chip.com/3090.html
 * Modular adaptation and changed parameters: see SOURCES.md.
 */
#ifndef DEMO_USART0_H
#define DEMO_USART0_H

#include "../demo_platform.h"

/* USART0 asynchronous 8N1, 115200 bps, TX0 -> IOA0 (逻辑A0), RX0 -> IOA1 (逻辑A1).
 *
 * Baud rate (spec 10.8.1 p.77):  baud = FOSC / ((BRGD + 1) * n)
 *   FOSC = 32 MHz (project HIRC clock, unchanged), n = TxSPD divider = 4 (TxSPD=00)
 *   BRGD = 0x044 = 68  ->  32 MHz / (69 * 4) = 115942.03 bps, +0.644 % vs 115200
 *   (the spec's own 32 MHz table lists the same row: n=4, BRGD=0x44, 0.64 %)
 * 8N1 frame: 8 data bits (TxL9/RxL9 = 0), no parity (none in USART0), and
 *   BRGD0H bit5 STOP1 = 1 -> 1 stop bit (reset value 0 means 2 stop bits).
 * Derivation and re-runnable check: agent_demo/tools/usart0_calc.py --check */
#define DEMO_USART0_BAUDRATE 115200u
#define DEMO_USART0_BRGD0H 0x20u
#define DEMO_USART0_BRGD0L 0x44u

/* Software receive queue depth (power of two). RxOVF happens after 3 unread
 * bytes in the hardware buffer, so bytes are moved out in the ISR. */
#define DEMO_USART0_RXQ_SIZE 16u
#define DEMO_USART0_TXQ_SIZE 16u

void Demo_Usart0_Init(void);
void Demo_Usart0_RxHandler(void);
/* Non-blocking: returns 1 when the byte was queued, 0 when the TX queue is full. */
unsigned char Demo_Usart0_WriteByte(unsigned char data);
/* Non-blocking: returns 1 and stores one received byte, 0 when nothing is pending. */
unsigned char Demo_Usart0_ReadByte(unsigned char *data);
/* Non-blocking: pushes the next queued byte into TX0REG when the register is free. */
void Demo_Usart0_Poll(void);
#endif
