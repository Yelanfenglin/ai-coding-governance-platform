"""Independent host checks against native manual semantics; no target SDK/board claim."""

import argparse
import configparser
import json
import os
import subprocess
import sys
from pathlib import Path

# Adapted from the existing independent USART acceptance verifier.
AGENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT))
from app.demos.firmware import code_mask, functions  # noqa: E402


def function(text, name):
    found = next(f for f in functions(code_mask(text)) if f.name == name)
    return text[found.start:found.end + 1]


def main(*, workspace: Path, seed: Path, out: Path, duty_register: int, compiler: str,
         baudrate: int = 115200):
    ROOT, BASE, HOST = workspace, out, out / "build"
    HOST.mkdir(parents=True, exist_ok=True)
    before = {p.relative_to(seed).as_posix(): p.read_bytes()
              for p in seed.rglob("*") if p.is_file()}
    unchanged = ["TC1_Work.c", "COMMON.H", "MASSESCHIP_DEFINE.H",
                 "agent_demo/modules/demo_pwm0.c", "agent_demo/modules/demo_pwm0.h"]
    if duty_register != 60:
        unchanged = unchanged[:3]
    for name in unchanged:
        assert (ROOT / name).read_bytes() == before[name], name
    old_main = before["main.C"].decode("utf-8").replace("\r\n", "\n")
    new_main = (ROOT / "main.C").read_text(encoding="utf-8")
    for name in ("SYSTEM_INIT", "Project_Work"):
        assert function(old_main, name) == function(new_main, name), name
    old, new = configparser.ConfigParser(), configparser.ConfigParser()
    old.read_string(before["M9P6362_TC1.prj"].decode())
    new.read(ROOT / "M9P6362_TC1.prj", encoding="utf-8")
    assert dict(old["Settings"]) == dict(new["Settings"])
    assert new.getint("File Count", "Count") == len(new["Project Files"])
    for p in (ROOT / "agent_demo/modules").glob("*usart*.c"):
        assert p.relative_to(ROOT).as_posix() in {
            n.replace("\\", "/") for n in new["Project Files"].values()
        }

    registers = {"OSCM", "GIEH", "GIEL", "MPPWM0", "T0CR", "T0CR2", "T1CR", "IOA0", "IOC1"}
    for port in "ABCD":
        registers.update(prefix + port for prefix in ("ANS", "PU", "PD", "IO", "OE"))
    for i in range(2):
        registers.update(f"TC{i}{s}" for s in ("CL", "CH", "EN", "IE", "IF", "IP"))
    registers.update(("TC1PRL", "TC1PRH", "ANSA0", "ANSA1", "OEA0", "OEA1", "PUA1"))
    for i in range(4):
        registers.update(f"{p}{i}" for p in ("ANSC", "OEC"))
        registers.update(f"PWM0{i}{s}" for s in ("CR", "D", "OE"))
    registers.update(("MPTX0", "MPRX0", "TX0CR", "RX0CR", "BRGD0H", "BRGD0L",
                      "TX0REG", "TX0IF", "TX0IE", "RX0IF", "RX0IE", "RX0IP", "FR0ER"))
    (HOST / "zc.h").write_text(
        "#define interrupt\n#define high_priority\n#define low_priority\n"
        "void test_watchdog(void);\n#define CLRWDT() test_watchdog()\n"
        "unsigned char host_read_rx(void);\n#define RX0REG host_read_rx()\n"
        + "\n".join(f"extern volatile unsigned char {r};" for r in sorted(registers)),
        encoding="utf-8",
    )
    (HOST / "registers.c").write_text(
        "\n".join(f"volatile unsigned char {r};" for r in sorted(registers)), encoding="utf-8",
    )
    (HOST / "check.c").write_text('''#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include "COMMON.H"
#include "agent_demo/demo_app.h"
#include "agent_demo/modules/demo_usart0.h"
void firmware_main(void); void int_h(void); void int_l(void);
extern unsigned int business_cycles;
static jmp_buf end;
static int passes;
static unsigned char hw_byte;
unsigned char host_read_rx(void) {
    unsigned char data = hw_byte;
    /* Reading the receive FIFO moves past this byte and its associated status.
       The next (empty) slot is clean. Native manual p80 samples FRER before RXREG. */
    RX0IF = 0; FR0ER = 0; RX0CR &= (unsigned char)~6u;
    return data;
}
void test_watchdog(void) {
    if (++passes == 4) longjmp(end, 1);
    TC1IF = 1; int_h();
}
static int expect_equal(const char *name, unsigned int actual, unsigned int expected) {
    if (actual != expected)
        fprintf(stderr, "%s: expected=%u actual=%u\\n", name, expected, actual);
    return actual == expected;
}
static int expect_between(const char *name, double actual, double low, double high) {
    int ok = actual >= low && actual <= high;
    if (!ok)
        fprintf(stderr, "%s: expected=[%g,%g] actual=%g\\n", name, low, high, actual);
    return ok;
}
int main(int argc, char **argv) {
    static const unsigned int n[] = {4,16,64,256};
    unsigned int divider, brgd;
    unsigned char data=0, accepted;
    double baud, error;
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    int ok;
    if (setjmp(end) == 0) firmware_main();
    divider = n[(TX0CR >> 1) & 3];
    brgd = ((BRGD0H & 3u) << 8) | BRGD0L;
    baud = 32000000.0 / ((brgd + 1) * divider);
    error = 100.0 * (baud / (double)EXPECTED_BAUDRATE - 1.0);
    if (mode == 0) {
        ok = (expect_equal("USART0.baudrate", DEMO_USART0_BAUDRATE, EXPECTED_BAUDRATE)
           && expect_equal("watchdog.passes", passes, 4)
           && expect_equal("business.cycles", business_cycles, 6)
           && expect_equal("business.IOC1_nonzero", IOC1 != 0, 1)
           && expect_equal("interrupt.GIEH", GIEH, 1)
           && expect_equal("TC1.PRL", TC1PRL, 0xE7)
           && expect_equal("TC1.PRH", TC1PRH, 0x03)
           && expect_equal("TC1.CR", T1CR, 0x0D)
           && expect_equal("PWM0.period", TC0CL, 199)
           && expect_equal("PWM0.duty", PWM00D, EXPECTED_DUTY_REGISTER)
           && expect_equal("PWM0.output_enabled", PWM00OE, 1)
           && expect_equal("TC0.mode", T0CR & 7, 3)
           && expect_equal("USART0.MPTX0", MPTX0, 0)
           && expect_equal("USART0.MPRX0", MPRX0, 1)
           && expect_equal("USART0.ANSA0", ANSA0, 0)
           && expect_equal("USART0.ANSA1", ANSA1, 0)
           && expect_equal("USART0.OEA0", OEA0, 1)
           && expect_equal("USART0.OEA1", OEA1, 0)
           && expect_equal("USART0.TX0CR_mask_B0", TX0CR & 0xB0, 0x80)
           && expect_equal("USART0.RX0CR_mask_90", RX0CR & 0x90, 0x80)
           && expect_equal("USART0.BRGD0H_bit5", (BRGD0H & 0x20) != 0, 1)
           && expect_between("USART0.baud_error_percent", error, -2.0, 2.0));
        printf("{\\"ok\\":%d,\\"baud\\":%.6f,\\"error_percent\\":%.6f,"
               "\\"brgd\\":%u,\\"divider\\":%u,\\"business_cycles\\":%u}\\n",
               ok,baud,error,brgd,divider,business_cycles);
    } else if (mode == 1) {
        hw_byte = 0x35; FR0ER = 0; RX0IF = 1; int_h(); int_l();
        accepted = Demo_Usart0_ReadByte(&data);
        ok = expect_equal("receive.accepted", accepted, 1)
          && expect_equal("receive.byte", data, 0x35)
          && expect_equal("receive.queue_empty", Demo_Usart0_ReadByte(&data), 0);
        printf("{\\"ok\\":%d,\\"received\\":%u}\\n",ok,data);
    } else if (mode == 2) {
        hw_byte = 0xE7; FR0ER = 1; RX0CR |= 2; RX0IF = 1; Demo_Usart0_RxHandler();
        accepted = Demo_Usart0_ReadByte(&data);
        ok = expect_equal("frame_error.accepted", accepted, 0)
          && expect_equal("frame_error.RX0IF", RX0IF, 0);
        printf("{\\"ok\\":%d,\\"bad_frame_accepted\\":%u}\\n",ok,accepted);
    } else {
        TX0IF = 0; TX0REG = 0xEE;
        accepted = Demo_Usart0_WriteByte(0x6A);
        ok = expect_equal("send.accepted", accepted, 1)
          && expect_equal("send.busy_register_unchanged", TX0REG, 0xEE);
        TX0IF = 1; Demo_Poll();
        ok = ok && expect_equal("send.transmitted", TX0REG, 0x6A);
        printf("{\\"ok\\":%d,\\"transmitted\\":%u}\\n",ok,TX0REG);
    }
    return ok ? 0 : 1;
}
''', encoding="utf-8")
    gcc = Path(compiler)
    env = {**os.environ, "PATH": str(gcc.parent) + os.pathsep + os.environ["PATH"]}
    common = [str(gcc), f"-DEXPECTED_DUTY_REGISTER={duty_register}",
              f"-DEXPECTED_BAUDRATE={baudrate}", "-std=c11", "-fcommon",
              "-Wall", "-Werror", "-Wno-main", "-I", str(HOST), "-I", str(ROOT)]
    commands = [
        [*common, "-Dmain=firmware_main", "-x", "c", "-c", str(ROOT / "main.C"),
         "-o", str(HOST / "firmware.o")],
        [*common, str(HOST / "firmware.o"),
         *[str(ROOT / name.replace("\\", "/")) for name in new["Project Files"].values()
           if Path(name).suffix.lower() == ".c" and Path(name).name.lower() != "main.c"],
         str(HOST / "registers.c"), str(HOST / "check.c"), "-o", str(HOST / "check.exe")],
    ]
    logs = []
    for command in commands:
        p = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
        logs.append({"command": command, "exit": p.returncode, "stdout": p.stdout,
                     "stderr": p.stderr})
        (BASE / "compile_checks.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
        assert p.returncode == 0, p.stderr
    checks = {}
    modes = ("configuration_and_old_behavior", "receive", "frame_error", "send")
    for mode, name in enumerate(modes):
        p = subprocess.run([str(HOST / "check.exe"), str(mode)], capture_output=True,
                           text=True, env=env, timeout=5)
        checks[name] = {"exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    main_poll = "Demo_Poll(" in function(new_main, "main")
    ok = all(row["exit"] == 0 for row in checks.values()) and main_poll
    failures = [{"check": name, "detail": row["stderr"] or row["stdout"]}
                for name, row in checks.items() if row["exit"] != 0]
    if not main_poll:
        failures.append({"check": "main_loop_calls_demo_poll",
                         "detail": "main: expected Demo_Poll() call; actual missing"})
    report = {"status": "passed" if ok else "failed", "failures": failures, "checks": checks,
              "unchanged": unchanged, "business_and_clock_preserved": True,
              "project_options_preserved": True, "target_compile": "not_run",
              "hardware": "not_run", "main_loop_calls_demo_poll": main_poll}
    (BASE / "independent_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duty-register", type=int, default=60)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--compiler", default=os.environ.get("CC", "gcc"))
    args = parser.parse_args()
    raise SystemExit(main(workspace=args.workspace, seed=args.seed, out=args.out,
                          duty_register=args.duty_register, compiler=args.compiler,
                          baudrate=args.baudrate))
