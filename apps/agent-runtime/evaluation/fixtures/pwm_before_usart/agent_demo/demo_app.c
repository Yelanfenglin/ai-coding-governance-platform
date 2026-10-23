#include "demo_app.h"
#include "modules/demo_pwm0.h"

void Demo_ModulesInit(void)
{
    unsigned char saved_gieh = GIEH;
    GIEH = 0;
    Demo_Pwm0_Init();
    GIEH = saved_gieh;
}

void Demo_EnableInterrupts(void)
{
}

void Demo_Poll(void)
{

}

void Demo_InterruptHigh(void)
{

}

void Demo_InterruptLow(void)
{
}
