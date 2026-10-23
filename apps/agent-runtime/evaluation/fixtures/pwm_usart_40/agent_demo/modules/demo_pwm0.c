/* Derived from MASSES CHIP M9P6362 Demo V1.07.
 * Original work: (C) COPYRIGHT 2025 Masses-Chip.
 * Official source: http://www.masses-chip.com/3090.html
 * Modular adaptation and changed parameters: see SOURCES.md.
 */
#include "demo_pwm0.h"

void Demo_Pwm0_Init(void)
{
//1、PWM0输出口设置
//PWM00复用口为C0
//PWM01复用口为C1
//PWM02复用口为C2
//PWM03复用口为C3
	ANSC0 = 0;
	OEC0 = 1;
	MPPWM0 = 0x20;
//2、TC0设置
	/* T0CR = 0x0B: TC0EN=0 (started last), TC0MOD=0 (8-bit, required by PWM0),
	 * TC0CKS=01 (FOSCH), TC0CKPS=011 (1:8).
	 * Same order as the official demo: configure T0CR/period/duty, start TC0 last. */
	T0CR = 0x0B;
    T0CR2 = 0;
	TC0CL = DEMO_PWM0_PERIOD;
	TC0CH = 0;
//3、PWM0设置
	PWM00CR = 0x80;
	PWM00D = DEMO_PWM0_DUTY;
	/* Start only after period and duty are loaded. 32 MHz / 8 = 4 MHz into TC0CH,
	 * TC0CL=199 -> 200 counts -> 20.000 kHz, PWM00D=80 -> 40.00 %.
	 * See agent_demo/tools/pwm0_calc.py. */
	TC0EN = 1;
    Demo_Pwm0_Start();
}

void Demo_Pwm0_Start(void)
{
    PWM00OE = 1;
}

void Demo_Pwm0_Stop(void)
{
    PWM00OE = 0;
}

void Demo_Pwm0_SetDuty(unsigned char ticks)
{
    PWM00D = ticks;
}
