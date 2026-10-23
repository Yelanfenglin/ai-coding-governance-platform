/* Derived from MASSES CHIP M9P6362 Demo V1.07.
 * Original work: (C) COPYRIGHT 2025 Masses-Chip.
 * Official source: http://www.masses-chip.com/3090.html
 * Modular adaptation and changed parameters: see SOURCES.md.
 */
#include "demo_timer1.h"

static volatile unsigned char demo_timer1_pending;

void Demo_Timer1_Init(void)
{
    ANSC1 = 0;
    OEC1 = 1;
    demo_timer1_pending = 0;
//1、TC1设置
    T1CR =0x0D; 			
    TC1CH = 0;				
    TC1CL = 0;				
	TC1PRL = (unsigned char)(DEMO_TIMER1_PERIOD & 0xFFu);
	TC1PRH = (unsigned char)(DEMO_TIMER1_PERIOD >> 8);
	TC1EN = 1;	
//2、高优先级中断设置 
	TC1IE = 1;
	TC1IF = 0;	
	TC1IP = 0;				
}

void Demo_Timer1_Interrupt (void)				
{
	if(TC1IF)
	{					
		TC1IF = 0;
		demo_timer1_pending = 1;
	}
}

void Demo_Timer1_Poll(void)
{
	if(demo_timer1_pending == 1)
	{
		demo_timer1_pending = 0;	
		IOC1 = ~IOC1;
	}
}

unsigned char Demo_Timer1_TakeTick(void)
{
    unsigned char pending;
    pending = demo_timer1_pending;
    demo_timer1_pending = 0;
    return pending;
}
