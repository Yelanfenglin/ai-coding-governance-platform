#!/usr/bin/env python3
"""Static/arithmetic checks for the M9P6362 ADC0 module (IOB7 / AD15).

No vendor compiler, no board, no scope: this only re-derives the register
numbers from the spec formulas and re-checks that the hand-written sources,
the .prj file table and demo-project.json still agree with them.

Spec sources (M9P6362 V1.16):
  §13.2 p.99  ADCON0: b7 ADEN / b6 ADS / b5 ADFM / b4:0 CHS[4:0]
              CHS 00000..11011 = AD0..AD27; ADFM 1 = right aligned
  §13.3 p.100 ADCON1: b7 AGEN / b6:4 ADCKS[2:0] / b3:2 VREMS / b1:0 VHS
              ADCKS 111 = Fcpu/128, VREMS 00 = VDD reference
  §13.5 p.102 ADH/ADL: ADFM=1 -> ADRES = {ADH[3:0], ADL[7:0]}
  §13.6 p.102 12-bit conversion time = 16 ADC clocks
  §1.3 p.15-17 IOB7 = AD15
  §6.6  p.51  ANSx: 0 = digital port, 1 = analog port
  §4.4  p.35  FCPU[2:0] = 110 -> 8T (instruction clock = FOSC/8)
"""
import argparse
import json
import os
import re
import sys

FOSC_HZ = 32_000_000          # HIRC 32 MHz, project option unchanged
CPU_DIV = 8                   # 8T option, FCPU=110
ADCKS = 0b111                 # ADCON1 bit[6:4]
ADC_CLOCK_LIMIT_HZ = 1_000_000
ADC_CLOCKS_PER_CONV = 16      # spec §13.6

CONV = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128}

# ADCON0 bit positions per §13.2 p.99 (bit6 is ADS, NOT ADFM).
ADCON0_ADEN_BIT = 7
ADCON0_ADS_BIT = 6
ADCON0_ADFM_BIT = 5


def derive():
    fcpu = FOSC_HZ // CPU_DIV
    # §13.3 says the divider input is "Fcpu"; the conservative check below also
    # evaluates the Fosc reading so the limit holds under either interpretation.
    lo = fcpu // CONV[ADCKS]
    hi = FOSC_HZ // CONV[ADCKS]
    lo_conv_us = ADC_CLOCKS_PER_CONV * 1e6 / lo
    hi_conv_us = ADC_CLOCKS_PER_CONV * 1e6 / hi
    adcon0 = (1 << ADCON0_ADEN_BIT) | (0 << ADCON0_ADS_BIT) | (1 << ADCON0_ADFM_BIT) | 15
    adcon1 = (ADCKS << 4) | 0x00         # AGEN=0, VREMS=00 (VDD), VHS=00
    return {
        "fcpu_hz": fcpu, "adcks": ADCKS, "divider": CONV[ADCKS],
        "adc_clock_low_hz": lo, "adc_clock_high_hz": hi,
        "conv_time_low_us": lo_conv_us, "conv_time_high_us": hi_conv_us,
        "adcon0": adcon0, "adcon1": adcon1,
        "channel": 15, "bits": 12,
    }


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def check(root):
    fails = []
    d = derive()
    if max(d["adc_clock_low_hz"], d["adc_clock_high_hz"]) > ADC_CLOCK_LIMIT_HZ:
        fails.append("ADC clock exceeds 1 MHz under some reading")
    if (d["adcon0"] & 0x1F) != 15:
        fails.append("CHS field is not AD15")
    if d["adcon0"] != 0xAF:
        fails.append("ADCON0 derivation is not 0xAF: %s" % hex(d["adcon0"]))
    if d["adcon1"] != 0x70:
        fails.append("ADCON1 derivation is not 0x70: %s" % hex(d["adcon1"]))

    hdr = read(os.path.join(root, "agent_demo/modules/demo_adc0.h"))
    src = read(os.path.join(root, "agent_demo/modules/demo_adc0.c"))
    app = read(os.path.join(root, "agent_demo/demo_app.c"))
    plat = read(os.path.join(root, "agent_demo/demo_platform.h"))
    prj = read(os.path.join(root, "M9P6362_TC1.prj"))
    js = json.loads(read(os.path.join(root, "demo-project.json")))

    # 1. register values in the sources match the derivation
    for pat in (r"ADCON0_VALUE\s+0xAFu", r"ADCON1_VALUE\s+0x70u", r"CHANNEL\s+15u"):
        if not re.search(pat, hdr):
            fails.append("header missing %s" % pat)
    if "ADCON0 = DEMO_ADC0_ADCON0_VALUE" not in src:
        fails.append("Init does not write ADCON0 from the macro")
    if "ADCON1 = DEMO_ADC0_ADCON1_VALUE" not in src:
        fails.append("Init does not write ADCON1 from the macro")

    # 2. Init must not start a conversion: no ADS=1 in Init
    init_body = src.split("unsigned char Demo_Adc0_Start")[0]
    if "ADS = 1" in init_body:
        fails.append("Init starts a conversion (ADS=1 found in Init)")
    if "ADS = 1" not in src:
        fails.append("Start never sets ADS")
    # ADCON0 bit decode: ADEN(b7)=1, ADS(b6)=0, ADFM(b5)=1
    if ((d["adcon0"] >> ADCON0_ADEN_BIT) & 1) != 1:
        fails.append("ADEN bit not set in ADCON0 value")
    if ((d["adcon0"] >> ADCON0_ADS_BIT) & 1) != 0:
        fails.append("ADS bit set in the Init ADCON0 value")
    if ((d["adcon0"] >> ADCON0_ADFM_BIT) & 1) != 1:
        fails.append("ADFM bit not set in ADCON0 value")

    # 3. no busy wait anywhere: no loop around ADS
    if re.search(r"(while|for|do)\s*\([^)]*ADS", src):
        fails.append("busy wait on ADS found")

    # 4. Start/Poll/Read must not call the delay helper
    after = src.split("unsigned char Demo_Adc0_Start")[1]
    if "Demo_Adc0_SettleDelay" in after:
        fails.append("Start/Poll/Read uses the delay helper")

    # 5. pin setup only for B7, no other port touched
    for pat in (r"ANSB7\s*=\s*1\s*;", r"PUB7\s*=\s*0\s*;", r"PDB7\s*=\s*0\s*;", r"OEB7\s*=\s*0\s*;"):
        if not re.search(pat, src):
            fails.append("missing pin setup %s" % pat)
    for m in re.finditer(r"\b(ANS[ACD][0-7]?|PU[ACD][0-7]?|PD[ACD][0-7]?|OE[ACD][0-7]?)\s*=", src):
        if not re.match(r"(ANS|PU|PD|OE)B7$", m.group(1)):
            fails.append("touches unrelated port register %s" % m.group(1))

    # 6. Read leaves *value untouched when no new result / consumes once
    if "*value = g_adc0_result;" not in src:
        fails.append("Read does not store the result")
    if src.count("*value = g_adc0_result;") != 1:
        fails.append("Read stores in more than one place")
    if src.index("if (g_adc0_state != DEMO_ADC0_READY)") > src.index("*value = g_adc0_result;"):
        fails.append("Read writes before checking for a new result")

    # 7. SFR names come from the device header chain (zc.h), not from this
    #    project's type-only MASSESCHIP_DEFINE.H: the module must pull them in
    #    through demo_platform.h and must not redefine ADCON0/ADS/AD itself.
    if '#include "zc.h"' not in plat:
        fails.append("demo_platform.h no longer includes the device header zc.h")
    if '#include "../demo_platform.h"' not in src:
        fails.append("demo_adc0.c does not include demo_platform.h (device SFRs)")
    for sym in ("ADCON0", "ADCON1", "ADS", "AD", "ANSB7", "PUB7", "PDB7", "OEB7"):
        if re.search(r"^#\s*define\s+%s\b" % sym, src, re.M):
            fails.append("module redefines device SFR %s" % sym)

    # 8. dispatch layer
    if "Demo_Adc0_Init();" not in app:
        fails.append("Demo_ModulesInit does not call Demo_Adc0_Init")
    if "Demo_Adc0_Poll();" not in app:
        fails.append("Demo_Poll does not call Demo_Adc0_Poll")

    # 9. .prj file table
    n = int(re.search(r"Count=(\d+)", prj).group(1))
    files = re.findall(r"^File\d+=(.+)$", prj.split("[Project Files]")[1].split("[Settings]")[0], re.M)
    if n != len(files):
        fails.append(".prj Count=%d but %d entries" % (n, len(files)))
    for want in ("agent_demo\\modules\\demo_adc0.h", "agent_demo\\modules\\demo_adc0.c"):
        if want not in files:
            fails.append(".prj missing %s" % want)

    # 10. demo-project.json (adc_clock_hz uses the checked Fcpu reading)
    mods = {m["id"]: m for m in js["modules"]}
    adc = mods.get("adc0")
    if adc is None:
        fails.append("demo-project.json has no adc0 module")
    else:
        if set(adc.get("resources", [])) != {"ADC", "IO:B7"}:
            fails.append("adc0 resources != {ADC, IO:B7}")
        if adc.get("init") != "Demo_Adc0_Init" or adc.get("poll") != "Demo_Adc0_Poll":
            fails.append("adc0 init/poll mismatch")
        a = adc.get("actual", {})
        if a.get("adcon0") != d["adcon0"] or a.get("adcon1") != d["adcon1"]:
            fails.append("adc0 actual adcon mismatch")
        if a.get("adc_clock_hz") != d["adc_clock_low_hz"]:
            fails.append("adc0 actual adc_clock_hz mismatch")
        if a.get("channel") != 15 or a.get("alignment") != "right" or a.get("reference") != "VDD":
            fails.append("adc0 actual channel/alignment/reference mismatch")
    for want in ("agent_demo/modules/demo_adc0.c", "agent_demo/modules/demo_adc0.h",
                 "agent_demo/tools/adc0_calc.py"):
        if want not in js.get("managed_files", []):
            fails.append("managed_files missing %s" % want)

    print("ADC clock: %d Hz (Fcpu=%d Hz reading) .. %d Hz (FOSC reading); limit %d Hz"
          % (d["adc_clock_low_hz"], d["fcpu_hz"], d["adc_clock_high_hz"], ADC_CLOCK_LIMIT_HZ))
    print("Conversion time (16 ADC clocks): %.1f us .. %.1f us"
          % (d["conv_time_low_us"], d["conv_time_high_us"]))
    print("ADCON0=0x%02X ADCON1=0x%02X channel=AD%d (IOB7)" % (d["adcon0"], d["adcon1"], d["channel"]))
    if fails:
        for f in fails:
            print("FAIL: " + f)
        return 1
    print("PASS: static checks (registers, no busy wait, pin setup, dispatch, .prj, manifest)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(args.root))
    for k, v in derive().items():
        print("%-18s %s" % (k, v))
