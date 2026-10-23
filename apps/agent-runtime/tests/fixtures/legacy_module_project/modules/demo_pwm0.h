/* Derived from MASSES CHIP M9P6362 Demo V1.07.
 * Original work: (C) COPYRIGHT 2025 Masses-Chip.
 * Official source: http://www.masses-chip.com/3090.html
 * Modular adaptation and changed parameters: see SOURCES.md.
 */
#ifndef DEMO_PWM0_H
#define DEMO_PWM0_H

#include "../demo_platform.h"

#define DEMO_PWM0_PERIOD 124u
#define DEMO_PWM0_DUTY 38u
void Demo_Pwm0_Init(void);
void Demo_Pwm0_Start(void);
void Demo_Pwm0_Stop(void);
/* ticks: high-level timer ticks, not percent; keep within the configured period. */
void Demo_Pwm0_SetDuty(unsigned char ticks);
#endif
