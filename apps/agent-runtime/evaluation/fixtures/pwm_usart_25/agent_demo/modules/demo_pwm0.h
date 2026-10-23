/* Derived from MASSES CHIP M9P6362 Demo V1.07.
 * Original work: (C) COPYRIGHT 2025 Masses-Chip.
 * Official source: http://www.masses-chip.com/3090.html
 * Modular adaptation and changed parameters: see SOURCES.md.
 */
#ifndef DEMO_PWM0_H
#define DEMO_PWM0_H

#include "../demo_platform.h"

/* 20 kHz / 25 % on C0: TC0 8-bit mode, FOSCH 32 MHz, TC0CKPS=011 (1:8).
 * period = TC0CL + 1 = 200 counts -> 32 MHz / (8 * 200) = 20.000 kHz (0 % error),
 * duty   = PWM00D / (TC0CL + 1) = 50 / 200 = 25.00 %  (200 duty steps).
 * Derivation and re-runnable check: agent_demo/tools/pwm0_calc.py --check */
#define DEMO_PWM0_PERIOD 199u
#define DEMO_PWM0_DUTY 50u
void Demo_Pwm0_Init(void);
void Demo_Pwm0_Start(void);
void Demo_Pwm0_Stop(void);
/* ticks: high-level timer ticks, not percent; keep within the configured period. */
void Demo_Pwm0_SetDuty(unsigned char ticks);
#endif
