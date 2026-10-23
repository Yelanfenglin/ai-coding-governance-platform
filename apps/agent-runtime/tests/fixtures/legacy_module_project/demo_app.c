#include "demo_app.h"
#include "modules/demo_pwm0.h"
#include "modules/demo_timer1.h"

void Demo_ModulesInit(void)
{
    Demo_Pwm0_Init();
    Demo_Timer1_Init();
}

void Demo_EnableInterrupts(void)
{
    GIEH = 1;
}

void Demo_Poll(void)
{
    Demo_Timer1_Poll();
}

void Demo_InterruptHigh(void)
{
    Demo_Timer1_Interrupt();
}

void Demo_InterruptLow(void)
{
}
