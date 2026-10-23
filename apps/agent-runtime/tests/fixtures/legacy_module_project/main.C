#include "demo_platform.h"
#include "demo_app.h"

void main(void)
{
    Demo_PlatformInit();
    Demo_ModulesInit();
    Demo_EnableInterrupts();
    while (1)
    {
        CLRWDT();
        Demo_Poll();
    }
}

void interrupt high_priority int_h(void)
{
    Demo_InterruptHigh();
}

void interrupt low_priority int_l(void)
{
    Demo_InterruptLow();
}
