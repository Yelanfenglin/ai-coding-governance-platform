#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M9P6362 TC0/PWM0 configuration calculator + verifier for this project.

Re-runnable:  python agent_demo/tools/pwm0_calc.py
              python agent_demo/tools/pwm0_calc.py --check   (also verify source files)

Model (M9P6362 spec V1.16, see SOURCES.md for section/page cites):
  * TC0CR bit6 TC0MOD = 0 -> 8-bit mode; TC0CL is the PERIOD register, TC0CH counts
    up; on TC0CH == TC0CL TC0 overflows and TC0CH restarts from 0  (§7.1 p.55).
    => timer period in prescaled clocks = TC0CL + 1   (counts 0..TC0CL)
  * TC0CKPS[2:0] prescaler = 1,2,4,8,16,32,64,128                        (§7.2 p.56)
  * TC0CKPO[2:0] (T0CR2, POSTscaler) = 1..8, "在原来预分频的基础上再分频" (§7.3 p.57).
    The spec text only states that it divides the prescaled clock further; this
    script does NOT assume how the postscaled overflow interacts with PWM output
    generation. Only what is read from the docs is used here: TC0CKPS (TC0CR)
    sets the prescaler that clocks TC0CH. This project keeps T0CR2 = 0, i.e. the
    postscaler stays at its reset/default 1:1 setting (same as the official PWM0
    demo), so the question is out of the calculation path entirely.
  * PWM00D is 8 bit; the waveform is high from count 0 up to PWM00D          (§7.7 p.58)
    => duty = PWM00D / (TC0CL + 1); the SPEC WAVEFORM EXAMPLE (§7.8 p.59,
       PWM00D=55h, TC0CL=FFh) confirms rise at 00H / fall at 55H i.e. /(TC0CL+1),
       NOT /256.
  * PWM00CR bit7 EN (requires 8-bit mode), bit6 OE, bit5 POL (0 = active high) (§7.6)
"""

from __future__ import print_function

FOSCH_HZ = 32000000  # HIRC 32 MHz (§4.1 p.33); not changed by this task
TARGET_F_HZ = 20000.0
TARGET_DUTY_PERCENT = 25.0
MAX_FREQ_ERROR_PERCENT = 0.1
MIN_DUTY_STEPS = 100

PRESCALERS = [1, 2, 4, 8, 16, 32, 64, 128]  # TC0CKPS[2:0] code == index
POSTSCALER = 1                              # T0CR2 = 0x00 -> TC0CKPO = 000 (1:1)

TC0CL_MIN = 0
TC0CL_MAX = 255   # 8-bit period register
PWM00D_MIN = 0    # 8-bit duty register
PWM00D_MAX = 255


def tc0cr_value(tc0en, tc0mod, tc0cks, tc0ckps):
    """T0CR: bit7 EN, bit6 MOD, bit5 reserved 0, bit[4:3] CKS, bit[2:0] CKPS."""
    return ((tc0en & 1) << 7) | ((tc0mod & 1) << 6) | ((tc0cks & 3) << 3) | (tc0ckps & 7)


def scan():
    """Enumerate every legal 8-bit-mode TC0 setting for the target frequency."""
    rows = []
    for ckps_code, presc in enumerate(PRESCALERS):
        tc0_clock = FOSCH_HZ / float(presc * POSTSCALER)  # Hz into TC0CH counter
        for tc0cl in range(TC0CL_MIN, TC0CL_MAX + 1):
            period_counts = tc0cl + 1                  # counts 0..TC0CL
            freq = tc0_clock / float(period_counts)
            freq_err = abs(freq - TARGET_F_HZ) / TARGET_F_HZ * 100.0
            ideal_duty = TARGET_DUTY_PERCENT / 100.0 * period_counts
            duty = int(round(ideal_duty))
            # best legal duty value inside a single period
            if duty > period_counts:
                duty = period_counts
            if duty > PWM00D_MAX:
                duty = PWM00D_MAX
            duty_percent = duty / float(period_counts) * 100.0
            duty_err = duty_percent - TARGET_DUTY_PERCENT
            rows.append({
                "ckps": ckps_code,
                "presc": presc,
                "t0cr": tc0cr_value(1, 0, 0b01, ckps_code),
                "tc0cl": tc0cl,
                "period_counts": period_counts,
                "freq": freq,
                "freq_err": freq_err,
                "duty": duty,
                "duty_percent": duty_percent,
                "duty_err": duty_err,
            })
    return rows


def select(rows):
    """Legal set = freq error within limit AND at least MIN_DUTY_STEPS duty steps."""
    legal = [r for r in rows
             if r["freq_err"] <= MAX_FREQ_ERROR_PERCENT
             and r["period_counts"] >= MIN_DUTY_STEPS]
    legal.sort(key=lambda r: (r["freq_err"], abs(r["duty_err"]), r["presc"]))
    return legal


def report(rows):
    legal = select(rows)
    print("M9P6362 TC0/PWM0 (PWM00 -> C0) configuration search")
    print("  FOSCH            = %d Hz  (HIRC 32 MHz, TC0CKS=01, unchanged)" % FOSCH_HZ)
    print("  target frequency = %.3f Hz, allowed error <= %.2f %%"
          % (TARGET_F_HZ, MAX_FREQ_ERROR_PERCENT))
    print("  target duty      = %.1f %%, need >= %d duty steps"
          % (TARGET_DUTY_PERCENT, MIN_DUTY_STEPS))
    print("  search space     = %d settings (prescaler x TC0CL 0..255, postscaler 1:1)"
          % len(rows))
    print("")

    exact = [r for r in legal if r["freq_err"] == 0.0]
    print("Exact-frequency candidates (error == 0.0 %%), sorted by duty error:")
    if not exact:
        print("  (none)")
    for r in exact:
        print("  TC0CKPS=%s (1:%-3d) TC0CL=%-3d period=%-3d counts  f=%9.3f Hz  "
              "PWM00D=%-3d duty=%6.2f %% (err %+0.2f pt)  T0CR=0x%02X"
              % (format(r["ckps"], "03b"), r["presc"], r["tc0cl"], r["period_counts"],
                 r["freq"], r["duty"], r["duty_percent"], r["duty_err"], r["t0cr"]))
    print("")

    # Best frequency match, then best duty match
    best = legal[0]
    print("Selected configuration:")
    print("  TC0CKPS[2:0] = %s -> prescaler 1:%d   (TC0CKS[1:0]=01 = FOSCH, TC0MOD=0 8-bit)"
          % (format(best["ckps"], "03b"), best["presc"]))
    print("  T0CR         = 0x%02X   (TC0MOD=0, TC0CKS=01, TC0CKPS=%s; TC0EN set afterwards)"
          % (best["t0cr"] & 0x7F, format(best["ckps"], "03b")))
    print("  T0CR2        = 0x00   (postscaler left at 1:1; its effect on PWM output is")
    print("                         not documented here, so it is kept at the default)")
    print("  TC0CL        = %d     (period = %d counts, i.e. counts 0..%d)"
          % (best["tc0cl"], best["period_counts"], best["tc0cl"]))
    print("  TC0CH        = 0")
    print("  PWM00CR      = 0x80   (PWM00EN=1, PWM00OE set separately, PWM00M=0 active high)")
    print("  PWM00D       = %d      (duty register)" % best["duty"])
    print("")
    print("Result:")
    print("  frequency      = %d / (%d * %d) = %.4f Hz  (target %.1f Hz, error %.4f %%)"
          % (FOSCH_HZ, best["presc"], best["period_counts"], best["freq"],
             TARGET_F_HZ, best["freq_err"]))
    print("  period time    = %.6f us" % (1e6 / best["freq"]))
    print("  duty           = %d/%d = %.4f %%  (target %.1f %%, error %+.4f percentage points)"
          % (best["duty"], best["period_counts"], best["duty_percent"],
             TARGET_DUTY_PERCENT, best["duty_err"]))
    print("  duty steps     = %d LSB per period (>= %d required)"
          % (best["period_counts"], MIN_DUTY_STEPS))
    print("  high time      = %.6f us, low time = %.6f us"
          % (best["duty"] / float(FOSCH_HZ / best["presc"]) * 1e6,
             (best["period_counts"] - best["duty"]) / float(FOSCH_HZ / best["presc"]) * 1e6))
    print("")
    print("Checks:")
    print("  [%s] frequency error %.4f %% <= %.2f %%" % (
        "PASS" if best["freq_err"] <= MAX_FREQ_ERROR_PERCENT else "FAIL",
        best["freq_err"], MAX_FREQ_ERROR_PERCENT))
    print("  [%s] duty steps %d >= %d" % (
        "PASS" if best["period_counts"] >= MIN_DUTY_STEPS else "FAIL",
        best["period_counts"], MIN_DUTY_STEPS))
    print("  [%s] TC0CL %d fits 8-bit period register" % (
        "PASS" if TC0CL_MIN <= best["tc0cl"] <= TC0CL_MAX else "FAIL", best["tc0cl"]))
    print("  [%s] PWM00D %d fits 8-bit duty register" % (
        "PASS" if PWM00D_MIN <= best["duty"] <= PWM00D_MAX else "FAIL", best["duty"]))
    print("  [%s] duty not truncated (PWM00D <= period)" % (
        "PASS" if best["duty"] <= best["period_counts"] else "FAIL"))
    return best


def check_sources(expected):
    """Verify the header macros and the module source agree with the calculation."""
    import os
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    hdr = os.path.join(root, "agent_demo", "modules", "demo_pwm0.h")
    src = os.path.join(root, "agent_demo", "modules", "demo_pwm0.c")
    print("")
    print("Source cross-check:")
    ok = True

    def read(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    htext = read(hdr)
    stext = read(src)

    def macro(name):
        m = re.search(r"#define\s+%s\s+(\d+)u" % name, htext)
        return int(m.group(1)) if m else None

    period = macro("DEMO_PWM0_PERIOD")
    duty = macro("DEMO_PWM0_DUTY")
    for label, got, want in (("DEMO_PWM0_PERIOD", period, expected["tc0cl"]),
                             ("DEMO_PWM0_DUTY", duty, expected["duty"])):
        good = got == want
        ok = ok and good
        print("  [%s] %s = %s (expected %d)" % ("PASS" if good else "FAIL", label, got, want))

    t0cr = re.search(r"T0CR\s*=\s*0x([0-9A-Fa-f]+)\s*;", stext)
    got_t0cr = int(t0cr.group(1), 16) if t0cr else None
    # Expected T0CR with TC0EN = 0 (timer started later by a separate bit write).
    want_t0cr = expected["t0cr"] & 0x7F
    good = got_t0cr == want_t0cr
    ok = ok and good
    print("  [%s] T0CR = 0x%02X (expected 0x%02X: MOD=0 8-bit, CKS=01 FOSCH, CKPS=%s, EN=0)"
          % ("PASS" if good else "FAIL", got_t0cr or 0, want_t0cr,
             format(expected["ckps"], "03b")))

    # TC0EN must be set by its own bit write placed after TC0CL and PWM00D,
    # matching the official demo order: configure first, then enable.
    en_write = re.search(r"\bTC0EN\s*=\s*1\s*;", stext)
    good = en_write is not None and got_t0cr is not None and not (got_t0cr & 0x80)
    ok = ok and good
    print("  [%s] TC0EN set by separate write, not inside the T0CR word" %
          ("PASS" if good else "FAIL"))
    for label, pat in (("TC0CL", r"\bTC0CL\s*="), ("PWM00D", r"\bPWM00D\s*=")):
        good = bool(en_write) and en_write.start() > re.search(pat, stext).start()
        ok = ok and good
        print("  [%s] TC0EN written after %s (configure, then enable)" % (
            "PASS" if good else "FAIL", label))

    t0cr2 = re.search(r"T0CR2\s*=\s*(0x[0-9A-Fa-f]+|0)\s*;", stext)
    got_t0cr2 = int(t0cr2.group(1), 0) if t0cr2 else None
    good = got_t0cr2 == 0
    ok = ok and good
    print("  [%s] T0CR2 = %s (T0CR2 = 0 keeps postscaler at default 1:1)" % (
        "PASS" if good else "FAIL", got_t0cr2))

    for name, want in (("MPPWM0", 0x20), ("PWM00CR", 0x80)):
        m = re.search(r"%s\s*=\s*0x([0-9A-Fa-f]+)\s*;" % name, stext)
        got = int(m.group(1), 16) if m else None
        good = got == want
        ok = ok and good
        print("  [%s] %s = 0x%02X (expected 0x%02X)" % (
            "PASS" if good else "FAIL", name, got or 0, want))
    return ok


def main(argv):
    rows = scan()
    best = report(rows)
    ok = True
    if "--check" in argv:
        ok = check_sources(best)
        print("")
        print("OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
