#include "demo_app.h"
#include "modules/demo_pwm0.h"
#include "modules/demo_usart0.h"
#include "modules/demo_adc0.h"

void Demo_ModulesInit(void)
{
    unsigned char saved_gieh = GIEH;
    GIEH = 0;
    Demo_Pwm0_Init();
    Demo_Usart0_Init();
    Demo_Adc0_Init();          /* enables ADC only, does not start a conversion */
    GIEH = saved_gieh;
}

void Demo_EnableInterrupts(void)
{
}

void Demo_Poll(void)
{
    /* 非阻塞：发送队列有数据且发送寄存器空闲时推入一个字节 */
    Demo_Usart0_Poll();
    /* 非阻塞：只推进已由 Demo_Adc0_Start() 发起的单次转换 */
    Demo_Adc0_Poll();
}

void Demo_InterruptHigh(void)
{
    /* RX0IP = 0，USART0 接收与 TC1 同在高优先级向量中，各自判断自己的标志 */
    Demo_Usart0_RxHandler();
}

void Demo_InterruptLow(void)
{
}
