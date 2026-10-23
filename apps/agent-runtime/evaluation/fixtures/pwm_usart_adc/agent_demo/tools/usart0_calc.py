#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M9P6362 USART0 (115200 8N1) configuration calculator + verifier for this project.

Re-runnable:  python agent_demo/tools/usart0_calc.py
              python agent_demo/tools/usart0_calc.py --check   (also verify source files)

Model (M9P6362 spec V1.16, see agent_demo/SOURCES.md for section/page cites):
  * Baud rate build: 目标波特率 = FOSC / ((BRGD + 1) x n)                  (§10.8.1 p.77)
    n = USART clock divider selected by TxSPD[1:0] (TXxCR bits[2:1]):
        00 -> 4, 01 -> 16, 10 -> 64, 11 -> 256                            (§10.2 p.74)
    FOSC here = HIRC 32 MHz (project clock, unchanged by this task).
  * The spec's own table (§10.8.1 p.77, Fosc = 32 MHz) lists 115200 bps ->
    divider 4, BRGD = 0x44, deviation 0.64 %.
  * BRGD is 10 bit wide: BRGDxH bits[1:0] = BRGD[9:8], BRGDxL bits[7:0] = BRGD[7:0]
                                                                            (§10.6/§10.7 p.76)
  * BRGDxH also carries the frame options: bit5 STOPx1 (0 = 2 stop bits, 1 = 1 stop
    bit) and bit7 SBYTEx, bit6 ODENx                                      (§10.6 p.76)
  * TXxCR bit7 TXxEN, bit5 TxSYNC (0 = async), bit4 TxL9 (0 = 8 data bits),
    bit6 TxMCLR is read-only "send register empty"                        (§10.2 p.74)
  * RXxCR bit7 RXxEN, bit4 RxL9 (0 = 8 data bits); RxOVF (bit2) and FRxER (bit1)
    are read-only status bits; reading RXxREG clears RXxIF/RxOVF  (§10.4/§10.5 p.75-76)
  * RX0IF is cleared automatically by reading RX0REG, TX0IF by writing TX0REG
                                                                     (§5.6 p.43)
  * Pin mapping MPTX0/MPRX0: PORTSEL[1:0] bits[5:4] (00=IOA, 01=IOB, 10=IOC,
    11=IOD), PINSEL[2:0] bits[2:0] (000=IOx0 ... 111=IOx7); USART TX/RX are
    全端口映射, so IOA0 (0x00) and IOA1 (0x01) are legal targets
                                                 (§6.9.1/§6.9.2 p.53, §1.2.6 note p.14)
"""

from __future__ import print_function

FOSC_HZ = 32000000          # HIRC 32 MHz; the baud formula in the spec uses FOSC
TARGET_BAUD = 115200
MAX_BAUD_ERROR_PERCENT = 2.0   # requirement from the task
SPEC_TABLE_ERROR_PERCENT = 0.64

# TxSPD[1:0] code -> divider n (§10.2 p.74)
SPD_DIVIDERS = [4, 16, 64, 256]
BRGD_MIN = 0
BRGD_MAX = 1023             # 10-bit register


def txcr_value(tx_en, tx_sync, tx_l9, spd_code, txd9=0):
    """TXxCR: bit7 EN, bit6 MCLR(R), bit5 SYNC, bit4 L9, bit3 SLAVE, bit[2:1] SPD, bit0 D9."""
    return (((tx_en & 1) << 7) | ((tx_sync & 1) << 5) | ((tx_l9 & 1) << 4)
            | ((spd_code & 3) << 1) | (txd9 & 1))


def brgdh_value(bgr, sbyte=0, oden=0, stop1=1):
    """BRGDxH: bit7 SBYTE, bit6 ODEN, bit5 STOP1, bits[1:0] BRGD[9:8]."""
    return (((sbyte & 1) << 7) | ((oden & 1) << 6) | ((stop1 & 1) << 5)
            | ((bgr >> 8) & 0x03))


def scan():
    """Enumerate every legal (TxSPD divider, 10-bit BRGD) pair for the target baud."""
    rows = []
    for spd_code, divider in enumerate(SPD_DIVIDERS):
        for brgd in range(BRGD_MIN, BRGD_MAX + 1):
            baud = FOSC_HZ / float((brgd + 1) * divider)
            err = abs(baud - TARGET_BAUD) / float(TARGET_BAUD) * 100.0
            rows.append({
                "spd": spd_code,
                "divider": divider,
                "brgd": brgd,
                "brgdh": brgdh_value(brgd, stop1=1),
                "brgdl": brgd & 0xFF,
                "txcr": txcr_value(1, 0, 0, spd_code),
                "baud": baud,
                "err": err,
            })
    return rows


def select(rows):
    legal = [r for r in rows if r["err"] <= MAX_BAUD_ERROR_PERCENT]
    # prefer the smallest divider (fastest internal baud clock -> most accurate
    # sampling), then the smallest error, then the lowest BRGD
    legal.sort(key=lambda r: (r["divider"], r["err"], r["brgd"]))
    return legal


def report(rows):
    legal = select(rows)
    print("M9P6362 USART0 (TX0 -> IOA0, RX0 -> IOA1) 8N1 configuration search")
    print("  FOSC             = %d Hz  (HIRC 32 MHz, unchanged)" % FOSC_HZ)
    print("  target baud      = %d bps, allowed error <= %.2f %%"
          % (TARGET_BAUD, MAX_BAUD_ERROR_PERCENT))
    print("  search space     = %d settings (4 dividers x BRGD 0..%d)"
          % (len(rows), BRGD_MAX))
    print("  formula          = FOSC / ((BRGD + 1) * n)                    [spec 10.8.1 p.77]")
    print("")

    best_by_div = {}
    for r in legal:
        if r["divider"] not in best_by_div:
            best_by_div[r["divider"]] = r
    print("Best BRGD per divider (error <= %.2f %%):" % MAX_BAUD_ERROR_PERCENT)
    for divider in SPD_DIVIDERS:
        r = best_by_div.get(divider)
        if r is None:
            print("  n = %-3d : no legal BRGD" % divider)
            continue
        print("  n = %-3d (TxSPD=%s) BRGD=0x%03X baud=%9.3f err=%+0.4f %%"
              % (divider, format(r["spd"], "02b"), r["brgd"], r["baud"], r["err"]))
    print("")

    best = legal[0]
    print("Selected configuration:")
    print("  TxSPD[1:0] = %s -> n = %d          (TX0CR bit[2:1], async, 8 data bits)"
          % (format(best["spd"], "02b"), best["divider"]))
    print("  BRGD       = 0x%03X = %d" % (best["brgd"], best["brgd"]))
    print("  BRGD0H     = 0x%02X  (SBYTE=0, ODEN=0, STOP1=1 -> 1 stop bit, BRGD[9:8]=%s)"
          % (best["brgdh"], format((best["brgd"] >> 8) & 3, "02b")))
    print("  BRGD0L     = 0x%02X  (BRGD[7:0])" % best["brgdl"])
    print("  TX0CR      = 0x%02X  (TX0EN=1, TxSYNC=0 async, TxL9=0 8 bit, SPD=%s)"
          % (best["txcr"], format(best["spd"], "02b")))
    print("  RX0CR      = 0x80     (RX0EN=1, RxL9=0 8 bit, SRxEN=0 async auto)")
    print("  MPTX0      = 0x00     (PORTSEL=00 IOA, PINSEL=000 IOA0)")
    print("  MPRX0      = 0x01     (PORTSEL=00 IOA, PINSEL=001 IOA1)")
    print("")
    print("Result:")
    print("  baud       = %d / ((0x%X + 1) * %d) = %.4f bps"
          % (FOSC_HZ, best["brgd"], best["divider"], best["baud"]))
    print("  deviation  = %+0.4f %% (spec table for 115200 @32 MHz: %+0.2f %%)"
          % (best["err"], -SPEC_TABLE_ERROR_PERCENT))
    print("  bit time   = %.4f us, one 8N1 frame = %.4f us"
          % (1e6 / best["baud"], 10e6 / best["baud"]))

    ok = True
    print("")
    print("Checks:")
    good = best["err"] <= MAX_BAUD_ERROR_PERCENT
    ok = ok and good
    print("  [%s] baud error %.4f %% <= %.2f %%"
          % ("PASS" if good else "FAIL", best["err"], MAX_BAUD_ERROR_PERCENT))
    good = 0 <= best["brgd"] <= BRGD_MAX
    ok = ok and good
    print("  [%s] BRGD 0x%X fits the 10-bit register" %
          ("PASS" if good else "FAIL", best["brgd"]))
    good = abs(best["err"] - SPEC_TABLE_ERROR_PERCENT) < 0.01
    ok = ok and good
    print("  [%s] matches the spec table row (115200 -> n=4, BRGD=0x44, |dev| 0.64 %%)"
          % ("PASS" if good else "FAIL"))
    # 8N1 frame check: 1 start + 8 data + 1 stop with STOP1 = 1
    good = bool(best["brgdh"] & 0x20)
    ok = ok and good
    print("  [%s] BRGD0H bit5 STOP1 = 1 -> 1 stop bit (8N1)" % ("PASS" if good else "FAIL"))
    return best, ok


def check_sources(expected):
    """Verify the module header/source agree with the calculation."""
    import os
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    hdr = os.path.join(root, "agent_demo", "modules", "demo_usart0.h")
    src = os.path.join(root, "agent_demo", "modules", "demo_usart0.c")
    app = os.path.join(root, "agent_demo", "demo_app.c")
    prj = os.path.join(root, "M9P6362_TC1.prj")
    print("")
    print("Source cross-check:")
    ok = True

    def read(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def result(label, good, detail=""):
        print("  [%s] %s%s" % ("PASS" if good else "FAIL", label,
                               ("  (%s)" % detail) if detail else ""))
        return good

    htext = read(hdr)
    stext = read(src)

    def hmacro(name, pattern=r"(\d+)u?"):
        m = re.search(r"#define\s+%s\s+%s" % (name, pattern), htext)
        return m.group(1) if m else None

    baud = hmacro("DEMO_USART0_BAUDRATE")
    ok &= result("DEMO_USART0_BAUDRATE = %s (expected %d)" % (baud, TARGET_BAUD),
                 baud is not None and int(baud) == TARGET_BAUD)

    want_h = "0x%02X" % expected["brgdh"]
    want_l = "0x%02X" % expected["brgdl"]
    got_h = hmacro("DEMO_USART0_BRGD0H", r"0x([0-9A-Fa-f]+)u?")
    got_l = hmacro("DEMO_USART0_BRGD0L", r"0x([0-9A-Fa-f]+)u?")
    ok &= result("DEMO_USART0_BRGD0H = 0x%s (expected %s)" % (got_h, want_h),
                 (got_h or "").upper() == want_h[2:].upper())
    ok &= result("DEMO_USART0_BRGD0L = 0x%s (expected %s)" % (got_l, want_l),
                 (got_l or "").upper() == want_l[2:].upper())

    # TX0CR written with TX0EN already set (single word: async, 8 bit, SPD=00)
    m = re.search(r"TX0CR\s*=\s*0x([0-9A-Fa-f]+)\s*;", stext)
    got = int(m.group(1), 16) if m else None
    ok &= result("TX0CR = %s (expected 0x%02X)" % (("0x%02X" % got) if got is not None else None,
                                                  expected["txcr"]),
                 got == expected["txcr"])

    # BRGD0H must be an exact write (|= would leave STOP1 at its reset 0 = 2 stop bits)
    ok &= result("BRGD0H written with '=' (not '|=') so STOP1 is defined",
                 re.search(r"BRGD0H\s*=\s*DEMO_USART0_BRGD0H\s*;", stext) is not None
                 or re.search(r"BRGD0H\s*=\s*0x[0-9A-Fa-f]+\s*;", stext) is not None)
    ok &= result("no 'BRGD0H |=' anywhere in the module",
                 re.search(r"BRGD0H\s*\|=", stext) is None)

    for name, want in (("MPTX0", 0x00), ("MPRX0", 0x01), ("RX0CR", 0x80)):
        m = re.search(r"%s\s*=\s*0x([0-9A-Fa-f]+)\s*;" % name, stext)
        got = int(m.group(1), 16) if m else None
        ok &= result("%s = %s (expected 0x%02X)" % (name, ("0x%02X" % got) if got is not None else None, want),
                     got == want)

    # Pull-up on RX (IOA1) only; TX stays push-pull output.
    ok &= result("PUA1 = 1 (RX pull-up), OEA0 = 1 / OEA1 = 0",
                 re.search(r"\bPUA1\s*=\s*1\s*;", stext) is not None
                 and re.search(r"\bOEA0\s*=\s*1\s*;", stext) is not None
                 and re.search(r"\bOEA1\s*=\s*0\s*;", stext) is not None)
    ok &= result("ANSA0 = 0 and ANSA1 = 0 (digital)",
                 re.search(r"\bANSA0\s*=\s*0\s*;", stext) is not None
                 and re.search(r"\bANSA1\s*=\s*0\s*;", stext) is not None)

    # Order: baud/frame registers first, then RX0IF cleared, then RX0IE, then RX0EN.
    def pos(pat):
        m = re.search(pat, stext)
        return m.start() if m else -1

    p_brgd = pos(r"BRGD0L\s*=")
    p_rxif = pos(r"RX0IF\s*=\s*0\s*;")
    p_rxie = pos(r"RX0IE\s*=\s*1\s*;")
    ok &= result("order: BRGD0L -> RX0IF=0 -> RX0IE=1",
                 -1 < p_brgd < p_rxif < p_rxie,
                 "offsets %d/%d/%d" % (p_brgd, p_rxif, p_rxie))
    ok &= result("RX0IP = 0 (USART0 RX on the high-priority vector, like TC1)",
                 re.search(r"\bRX0IP\s*=\s*0\s*;", stext) is not None)

    # Non-blocking API: no busy-wait loops on the send flag, no delay calls.
    ok &= result("no while-loop busy wait in the module",
                 re.search(r"\bwhile\s*\(", stext) is None)
    ok &= result("no blocking delay function in the module",
                 re.search(r"Delay|delay_ms|Delay_50ms", stext) is None)
    ok &= result("TX0IF used as the send-register-full flag (read only)",
                 re.search(r"TX0IF", stext) is not None
                 and re.search(r"TX0IF\s*=", stext) is None)

    # spec 10.8.3 p.80 step order: STEP4 check FRxER, STEP6 read RXxREG.
    # Reading RXxREG first advances the buffer, so FRxER/D9 would describe the
    # NEXT byte -- a bad frame would be accepted. FR0ER must therefore be
    # sampled before the first RX0REG read inside the handler.
    handler = stext[stext.index("Demo_Usart0_RxHandler(void)"):] if "Demo_Usart0_RxHandler(void)" in stext else ""
    p_frer = handler.find("FR0ER")
    p_rxreg = handler.find("RX0REG")
    ok &= result("RxHandler samples FR0ER before reading RX0REG (spec 10.8.3 STEP4 -> STEP6)",
                 p_frer >= 0 and p_rxreg >= 0 and p_frer < p_rxreg,
                 "offsets FR0ER=%d RX0REG=%d" % (p_frer, p_rxreg))
    ok &= result("RxHandler reads RX0REG (clears RX0IF / RxOVF)",
                 p_rxreg >= 0)

    # Frames with FR0ER set must not reach the software queue.
    guard = re.search(r"if\s*\(\s*FR0ER\s*==\s*0\s*\)(.*?)\}\s*else", handler, re.S)
    ok &= result("queue push only inside 'if (FR0ER == 0)' block",
                 guard is not None and "rx_queue[" in guard.group(1))

    # Hooks into the existing project: module wiring AND the real main.C/int_h.
    atext = read(app)
    p_pwm = re.search(r"Demo_Pwm0_Init\s*\(\s*\)\s*;", atext)
    p_usart = re.search(r"Demo_Usart0_Init\s*\(\s*\)\s*;", atext)
    ok &= result("Demo_Usart0_Init() called from Demo_ModulesInit()",
                 p_usart is not None and p_pwm is not None
                 and p_usart.start() > p_pwm.start())
    ok &= result("Demo_InterruptHigh() called from Demo_InterruptHigh hooks",
                 re.search(r"Demo_Usart0_RxHandler\s*\(\s*\)\s*;", atext) is not None)
    ok &= result("Demo_Poll() forwards to Demo_Usart0_Poll()",
                 re.search(r"Demo_Poll\s*\(\s*void\s*\)[^}]*Demo_Usart0_Poll\s*\(\s*\)\s*;",
                           atext, re.S) is not None)
    ok &= result("GIEL stays untouched in demo_app.c (TC1 1 ms event on GIEH)",
                 re.search(r"GIEL\s*=", atext) is None)
    ok &= result("demo_platform.h keeps the original zc.h include path",
                 read(os.path.join(root, "agent_demo", "demo_platform.h")).strip()
                 == '#ifndef DEMO_PLATFORM_H\n#define DEMO_PLATFORM_H\n#include "zc.h"\n'
                    '#include "../MASSESCHIP_DEFINE.H"\n#endif')

    mtext = read(os.path.join(root, "main.C"))
    loop = re.search(r"while\s*\(\s*1\s*\)\s*\{(.*?)\}", mtext, re.S)
    body = loop.group(1) if loop else ""
    ok &= result("main.C main loop calls Demo_Poll()", "Demo_Poll" in body)
    ok &= result("main.C main loop keeps CLRWDT() and TC1_Work() and Project_Work()",
                 all(k in body for k in ("CLRWDT", "TC1_Work", "Project_Work")))
    isr = re.search(r"void\s+interrupt\s+high_priority\s+int_h\s*\(\s*void\s*\)\s*\{(.*?)\n\}",
                    mtext, re.S)
    ibody = isr.group(1) if isr else ""
    ok &= result("main.C int_h() calls Demo_InterruptHigh()",
                 "Demo_InterruptHigh" in ibody)
    ok &= result("main.C int_h() keeps TC1_Interrupt() first",
                 "TC1_Interrupt" in ibody
                 and ibody.index("TC1_Interrupt") < ibody.index("Demo_InterruptHigh")
                 if "Demo_InterruptHigh" in ibody else False)
    lbody = re.search(r"void\s+interrupt\s+low_priority\s+int_l\s*\(\s*void\s*\)\s*\{(.*?)\n\}",
                      mtext, re.S)
    ok &= result("main.C int_l() still empty (GIEL unused)", lbody is not None)
    ok &= result("main.C Project_Work() body unchanged",
                 re.search(r"void\s+Project_Work\s*\(\s*void\s*\)\s*\{\s*"
                           r"business_cycles\s*\+=\s*2\s*;[^}]*\}", mtext, re.S) is not None)
    ok &= result("main.C SYSTEM_INIT() keeps OSCM = 0x00 and GIEH/GIEL = 0",
                 "OSCM = 0x00;" in mtext and "GIEH = 0;" in mtext and "GIEL = 0;" in mtext)

    # Project file list must mention the new sources.
    ptext = read(prj)
    n_files = re.search(r"\[File Count\]\s*\n\s*Count\s*=\s*(\d+)", ptext)
    listed = set(re.findall(r"File\d+\s*=\s*([^\r\n]+)", ptext))
    listed = set(x.strip() for x in listed)
    ok &= result("M9P6362_TC1.prj lists demo_usart0.c and demo_usart0.h",
                 any("demo_usart0.c" in x for x in listed)
                 and any("demo_usart0.h" in x for x in listed),
                 "Count=%s" % (n_files.group(1) if n_files else "?"))
    ok &= result("M9P6362_TC1.prj still lists the PWM0 / app files",
                 all(any(f in x for x in listed)
                     for f in ("demo_pwm0.c", "demo_pwm0.h", "demo_app.c",
                               "TC1_Work.c", "main.C")))
    return ok


def main(argv):
    rows = scan()
    best, ok = report(rows)
    if "--check" in argv:
        ok = check_sources(best) and ok
        print("")
        print("OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
