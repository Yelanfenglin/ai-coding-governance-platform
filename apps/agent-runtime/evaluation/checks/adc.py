"""ADC host oracle migrated from the accepted development experiment.

Only the host SFR shim, additional ADC test modes and compiler inputs are extended.
The original predicates remain intact; failed comparisons now explain expected
and observed values. Historical experiment scripts are kept unchanged.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HEADER = r'''
#ifndef HOST_ADC_SFRS_H
#define HOST_ADC_SFRS_H
typedef union { unsigned char value; struct {
 unsigned char b0:1,b1:1,b2:1,b3:1,b4:1,b5:1,b6:1,b7:1;
} bits; } host_sfr;
extern volatile host_sfr host_adcon0, host_ansb, host_oeb, host_pub, host_pdb;
extern volatile union host_ad_result { unsigned short value; struct {
 unsigned char low,high;
} bytes; } host_ad;
#define ADCON0 host_adcon0.value
#define ADEN host_adcon0.bits.b7
#define ADS host_adcon0.bits.b6
#define ADFM host_adcon0.bits.b5
#define ANSB host_ansb.value
#define ANSB7 host_ansb.bits.b7
#define OEB host_oeb.value
#define OEB7 host_oeb.bits.b7
#define PUB host_pub.value
#define PUB7 host_pub.bits.b7
#define PDB host_pdb.value
#define PDB7 host_pdb.bits.b7
#define AD host_ad.value
#define ADH host_ad.bytes.high
#define ADL host_ad.bytes.low
#define NOP() ((void)0)
#endif
'''
REGISTERS = '''#include "zc.h"
volatile host_sfr host_adcon0, host_ansb, host_oeb, host_pub, host_pdb;
volatile union host_ad_result host_ad;
'''
ADC_TEST = r'''
#include <stdio.h>
#include "COMMON.H"
#include "agent_demo/modules/demo_adc0.h"
static int expect_adc(const char *name, unsigned int actual, unsigned int expected) {
 if (actual != expected)
   fprintf(stderr, "%s: expected=%u actual=%u\n", name, expected, actual);
 return actual == expected;
}
static int expect_adc_divider(unsigned int actual) {
 if (actual < 2)
   fprintf(stderr, "ADC.Init.ADCKS: expected>=2 actual=%u\n", actual);
 return actual >= 2;
}
int adc_test(int mode) {
 unsigned int value = 0xBEEF;
 unsigned int samples[] = {0, 0x800, 0xFFF};
 unsigned int n;
 int ok;
 ANSB = 0x15; OEB = 0xDA; PUB = 0xFF; PDB = 0xFF;
 ADCON0 = 0; ADCON1 = 0; ADCON2 = 0;
 Demo_Adc0_Init();
 ok = (expect_adc("ADC.Init.ADEN", ADEN, 1)
       && expect_adc("ADC.Init.ADFM", ADFM, 1)
       && expect_adc("ADC.Init.ADS", ADS, 0)
       && expect_adc("ADC.Init.CHS", ADCON0 & 31, 15)
       && expect_adc("ADC.Init.VREMS", (ADCON1 >> 2) & 3, 0)
       && expect_adc_divider((ADCON1 >> 4) & 7)
       && expect_adc("ADC.Init.ANSB", ANSB, 0x95)
       && expect_adc("ADC.Init.OEB", OEB, 0x5A)
       && expect_adc("ADC.Init.PUB", PUB, 0x7F)
       && expect_adc("ADC.Init.PDB", PDB, 0x7F));
 if (mode == 4) {
   ok = ok && expect_adc("ADC.Read.no_result", Demo_Adc0_Read(&value), 0)
           && expect_adc("ADC.Read.output_unchanged", value, 0xBEEF);
 } else if (mode == 5) {
   ok = ok && expect_adc("ADC.Start.accepted", Demo_Adc0_Start(), 1)
           && expect_adc("ADC.Start.ADS", ADS, 1);
   ok = ok && expect_adc("ADC.Start.busy_rejected", Demo_Adc0_Start(), 0)
           && expect_adc("ADC.Start.busy_ADS", ADS, 1);
   for(n=0;n<1000;n++) Demo_Adc0_Poll();
   ok = ok && expect_adc("ADC.Poll.busy_ADS", ADS, 1)
           && expect_adc("ADC.Read.busy_no_result", Demo_Adc0_Read(&value), 0)
           && expect_adc("ADC.Read.busy_output_unchanged", value, 0xBEEF);
 } else {
   for(n=0;n<3;n++) {
     ok = ok && expect_adc("ADC.Start.accepted", Demo_Adc0_Start(), 1)
             && expect_adc("ADC.Start.ADS", ADS, 1);
     AD = (unsigned short)samples[n]; ADS = 0;
     Demo_Adc0_Poll();
     ok = ok && expect_adc("ADC.Read.ready", Demo_Adc0_Read(&value), 1)
             && expect_adc("ADC.Read.sample", value, samples[n]);
     value = 0xBEEF;
     ok = ok && expect_adc("ADC.Read.consumed", Demo_Adc0_Read(&value), 0)
             && expect_adc("ADC.Read.consumed_output_unchanged", value, 0xBEEF)
             && expect_adc("ADC.Poll.no_auto_restart", ADS, 0);
   }
 }
 printf("{\"ok\":%d,\"mode\":%d,\"adcon0\":%u,\"adcon1\":%u}\n",
        ok,mode,(unsigned int)ADCON0,(unsigned int)ADCON1);
 return ok ? 0 : 1;
}
'''


def build_checker():
    original = (ROOT / "evaluation/checks/firmware.py").read_text(encoding="utf-8")
    source = original

    def patch(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after, 1)

    patch('    (HOST / "zc.h").write_text(',
          '    registers.difference_update({"ANSB","OEB","PUB","PDB"})\n'
          '    registers.update({"ADCON1","ADCON2","ADIE","ADIF"})\n'
          '    (HOST / "zc.h").write_text(')
    patch('        "#define interrupt\\n#define high_priority\\n#define low_priority\\n"',
          '        EXTRA_HEADER + '
          '"#define interrupt\\n#define high_priority\\n#define low_priority\\n"')
    patch('        "\\n".join(f"volatile unsigned char {r};" for r in sorted(registers)),',
          '        EXTRA_REGISTERS + '
          '"\\n".join(f"volatile unsigned char {r};" for r in sorted(registers)),')
    patch('#include "agent_demo/modules/demo_usart0.h"',
          '#include "agent_demo/modules/demo_usart0.h"\nint adc_test(int mode);')
    patch('    int mode = argc > 1 ? atoi(argv[1]) : 0;',
          '    int mode = argc > 1 ? atoi(argv[1]) : 0;\n    if (mode >= 4) return adc_test(mode);')
    patch('    gcc = Path(compiler)',
          '    (HOST / "adc_check.c").write_text(EXTRA_ADC_TEST, encoding="utf-8")\n'
          '    gcc = Path(compiler)')
    patch('         str(HOST / "registers.c"), str(HOST / "check.c"),',
          '         str(HOST / "adc_check.c"), str(HOST / "registers.c"), str(HOST / "check.c"),')
    patch('    modes = ("configuration_and_old_behavior", "receive", "frame_error", "send")',
          '    modes = ("configuration_and_old_behavior", "receive", "frame_error", "send",\n'
          '             "adc_init", "adc_busy_nonblocking", "adc_results")')
    namespace = {"__file__": str(ROOT / "evaluation/checks/firmware.py"),
                 "__name__": "adc_oracle", "EXTRA_HEADER": HEADER,
                 "EXTRA_REGISTERS": REGISTERS, "EXTRA_ADC_TEST": ADC_TEST}
    exec(compile(source, "frozen_firmware_oracle_with_adc.py", "exec"), namespace)
    return namespace["main"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code = build_checker()(workspace=args.workspace, seed=args.seed, out=args.out,
                           duty_register=60, compiler=os.environ.get("CC", "gcc"))
    manifest = json.loads((args.workspace / "demo-project.json").read_text(encoding="utf-8"))
    adc = next(item for item in manifest["modules"] if item["id"] == "adc0")
    assert adc["init"] == "Demo_Adc0_Init" and adc["poll"] == "Demo_Adc0_Poll"
    assert {"ADC", "IO:B7"} <= set(adc["resources"])
    raise SystemExit(code)
