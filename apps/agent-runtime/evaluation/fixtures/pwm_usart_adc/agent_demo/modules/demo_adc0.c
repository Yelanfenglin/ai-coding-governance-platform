/* Derived from MASSES CHIP M9P6362 Demo V1.07.
 * Original work: (C) COPYRIGHT 2025 Masses-Chip.
 * Official source: http://www.masses-chip.com/3090.html
 * Modular adaptation and changed parameters: see SOURCES.md.
 */
#include "demo_adc0.h"
#include "../demo_platform.h"

#define DEMO_ADC0_IDLE   0u
#define DEMO_ADC0_BUSY   1u
#define DEMO_ADC0_READY  2u

static unsigned char  g_adc0_state;
static unsigned int   g_adc0_result;

/* Finite init-only delay (16 us units at 32 MHz / 8T), never used in Start/Poll/Read. */
static void Demo_Adc0_SettleDelay(unsigned char num16us)
{
    volatile unsigned int i, j;
    for (i = 0; i < num16us; i++)
    {
        for (j = 0; j < 3; j++)
        {
        }
    }
}

void Demo_Adc0_Init(void)
{
    /* IO B7: analog input, digital input disabled, pull-up and pull-down off.
     * ANSB7=1 per spec §6.6 (1 = analog port). Only B7 is touched. */
    ANSB7 = 1;
    PUB7  = 0;
    PDB7  = 0;
    OEB7  = 0;

    /* §13.3 ADCON1: AGEN=0, ADCKS=111 (Fcpu/128), VREMS=00 (VDD), VHS=00 */
    ADCON1 = DEMO_ADC0_ADCON1_VALUE;
    /* §13.2 ADCON0: bit7 ADEN=1, bit6 ADS=0, bit5 ADFM=1 (right aligned),
     * bits[4:0] CHS=01111 (AD15). Init only enables the ADC; it must not
     * start a conversion, so bit6 (ADS) stays 0. */
    ADCON0 = DEMO_ADC0_ADCON0_VALUE;

    g_adc0_state  = DEMO_ADC0_IDLE;
    g_adc0_result = 0u;

    /* Allow the reference / input switch to settle (finite, init only). */
    Demo_Adc0_SettleDelay(DEMO_ADC0_SETTLE_16US);
}

unsigned char Demo_Adc0_Start(void)
{
    /* No restart while converting and no restart while an unread result is
     * pending: a second Start must never overwrite data the caller has not
     * taken yet. */
    if (g_adc0_state != DEMO_ADC0_IDLE)
    {
        return 0;
    }
    g_adc0_state = DEMO_ADC0_BUSY;
    /* ADS=1 starts one conversion; hardware clears it when done (§13.2).
     * Returns immediately, no wait for completion. */
    ADS = 1;
    return 1;
}

void Demo_Adc0_Poll(void)
{
    if (g_adc0_state != DEMO_ADC0_BUSY)
    {
        return;
    }
    /* Progress the pending request only; no busy wait, no automatic restart. */
    if (ADS != 0)
    {
        return;
    }
    g_adc0_result = (unsigned int)(AD & 0x0FFFu);   /* ADFM=1: ADRES = {ADH[3:0], ADL[7:0]} */
    g_adc0_state  = DEMO_ADC0_READY;
}

unsigned char Demo_Adc0_Read(unsigned int *value)
{
    if (value == 0)
    {
        return 0;
    }
    if (g_adc0_state != DEMO_ADC0_READY)
    {
        return 0;                 /* no new result: *value untouched */
    }
    *value = g_adc0_result;
    g_adc0_state = DEMO_ADC0_IDLE;
    return 1;                     /* consumed exactly once */
}
