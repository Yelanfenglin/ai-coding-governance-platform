/* Derived from MASSES CHIP M9P6362 Demo V1.07.
 * Original work: (C) COPYRIGHT 2025 Masses-Chip.
 * Official source: http://www.masses-chip.com/3090.html
 * Modular adaptation and changed parameters: see SOURCES.md.
 */
#ifndef DEMO_TIMER1_H
#define DEMO_TIMER1_H

#include "../demo_platform.h"

#define DEMO_TIMER1_PERIOD 9999u
void Demo_Timer1_Init(void);
void Demo_Timer1_Interrupt(void);
unsigned char Demo_Timer1_TakeTick(void);
/* TakeTick coalesces pending interrupts, as in the vendor's flag-based example. */
void Demo_Timer1_Poll(void);
#endif
