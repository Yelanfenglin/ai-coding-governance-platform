#ifndef DEMO_ADC0_H
#define DEMO_ADC0_H

/* VDD reference, 12-bit right-justified, channel AD15 (IOB7).
 * ADCON1 = 0x70: AGEN=0, ADCKS=111 (ADC clock = src/128), VREMS=00 (VDD), VHS=00.
 * ADCON0 = 0xAF: ADEN=1, ADS=0 (Init must NOT start), ADFM=1 (right), CHS=0x0F (AD15).
 * See agent_demo/tools/adc0_calc.py for the runnable derivation. */
#define DEMO_ADC0_ADCON0_VALUE 0xAFu
#define DEMO_ADC0_ADCON1_VALUE 0x70u
#define DEMO_ADC0_CHANNEL      15u   /* CHS[4:0] = 01111 -> AD15 */

/* Finite stabilisation delay for the ADC block/pin switch, in 16 us units
 * at 32 MHz / 8T (same loop shape as the official ADC_Delay16us()). */
#define DEMO_ADC0_SETTLE_16US  8u

void Demo_Adc0_Init(void);
unsigned char Demo_Adc0_Start(void);
void Demo_Adc0_Poll(void);
unsigned char Demo_Adc0_Read(unsigned int *value);

#endif
