#ifndef DEMO_ADC0_H
#define DEMO_ADC0_H

/* VDD reference, 12-bit right-justified, channel AD15 (IOB7).
 * ADCON0 bit layout per spec §13.2 p.99: bit7 ADEN, bit6 ADS, bit5 ADFM, bits[4:0] CHS.
 * ADCON0 = 0xAF: ADEN=1 (b7), ADS=0 (b6, Init must NOT start), ADFM=1 (b5, right), CHS=0x0F (AD15).
 * ADCON1 = 0x70: AGEN=0 (b7), ADCKS=111 (b6:4 -> Fcpu/128), VREMS=00 (b3:2, VDD), VHS=00 (b1:0).
 * ADC clock = 4 MHz/128 = 31250 Hz (Fcpu reading) or 32 MHz/128 = 250 kHz (Fosc reading);
 * both are within the 1 MHz limit, 12-bit conversion = 16 ADC clocks (§13.6 p.102).
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
