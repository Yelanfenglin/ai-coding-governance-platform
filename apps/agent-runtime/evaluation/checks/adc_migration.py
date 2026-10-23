"""One ADC migration case: shared C behavior contract, chip-specific configuration.

Runs actual source/target module C with host SFR aliases and injected conversion
completion. The target main and original TC1 code are also compiled and exercised.
These are task checks, not a migration engine or target-device emulator.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.demos.firmware import code_mask, functions  # noqa: E402

SEED = Path(__file__).resolve().parents[1] / "fixtures" / "chip_migration_adc"


def text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("gb18030")


def function_body(path: Path, name: str) -> str:
    source = code_mask(text(path))
    function = next(item for item in functions(source) if item.name == name)
    return re.sub(r"\s+", "", source[function.start:function.end + 1])


def host_files() -> tuple[str, str]:
    lines = ["#ifndef HOST_ZC_H", "#define HOST_ZC_H", "#define interrupt",
             "#define high_priority", "#define low_priority", "#define bit unsigned char",
             "void host_watchdog(void);", "#define CLRWDT() host_watchdog()",
             "#define NOP() ((void)0)",
             "typedef union { unsigned char value; struct { unsigned char "
             "b0:1,b1:1,b2:1,b3:1,b4:1,b5:1,b6:1,b7:1; } bits; } host_sfr;"]
    sfrs = ["ADCON0", "ADCON1", "ADCON2"]
    sfrs += [prefix + port for prefix in ("ANS", "OE", "PU", "PD", "IO") for port in "ABC"]
    for name in sfrs:
        lines += [f"extern volatile host_sfr h_{name};", f"#define {name} h_{name}.value"]
        if not name.startswith("ADCON"):
            lines += [f"#define {name}{i} h_{name}.bits.b{i}" for i in range(8)]
    ordinary = ["OSCM", "GIEH", "GIEL", "T1CR", "TC1CH", "TC1CL", "TC1PRH", "TC1PRL",
                "TC1EN", "TC1IE", "TC1IF", "TC1IP"]
    lines += ["#define ADEN h_ADCON0.bits.b7", "#define ADS h_ADCON0.bits.b6",
              "#define ADFM h_ADCON0.bits.b5",
              "extern volatile union host_adc_result { unsigned short value; "
              "struct { unsigned char low, high; } bytes; } h_result;",
              "#define AD h_result.value", "#define ADL h_result.bytes.low",
              "#define ADH h_result.bytes.high"]
    lines += [f"extern volatile unsigned char {name};" for name in ordinary]
    lines += ["#endif"]
    registers = ['#include "zc.h"', "volatile union host_adc_result h_result;"]
    registers += [f"volatile host_sfr h_{name};" for name in sfrs]
    registers += [f"volatile unsigned char {name};" for name in ordinary]
    return "\n".join(lines), "\n".join(registers)


BEHAVIOR = r'''
#include <stdio.h>
#include "zc.h"
#include "module.h"
_Static_assert(_Generic(&Demo_Adc0_Init, void (*)(void):1, default:0), "Init API changed");
_Static_assert(_Generic(&Demo_Adc0_Start, unsigned char (*)(void):1, default:0),
               "Start API changed");
_Static_assert(_Generic(&Demo_Adc0_Poll, void (*)(void):1, default:0), "Poll API changed");
_Static_assert(_Generic(&Demo_Adc0_Read, unsigned char (*)(unsigned int *):1, default:0),
               "Read API changed");
#define EQ(label,got,want) do { unsigned int g=(got),w=(want); if(g!=w) { \
 fprintf(stderr,"%s: expected=%u actual=%u\n",label,w,g); return 1; } } while(0)
void host_watchdog(void) {}
int main(void) {
 unsigned int value=0xBEEF, i, j;
 const unsigned int samples[]={0,1,2048,4095,123,3000,0xF123};
 ANSB=0x05; OEB=PUB=PDB=0xFF; GIEH=GIEL=1; T1CR=12; TC1EN=1;
 Demo_Adc0_Init();
 EQ("init.enable",ADEN,1); EQ("init.right_alignment",ADFM,1); EQ("init.no_start",ADS,0);
 EQ("init.channel",ADCON0&31,EXPECT_CHANNEL);
 EQ("init.divider",(ADCON1>>4)&7,EXPECT_DIVIDER);
 EQ("init.VDD_reference",ADCON1&0x8F,0);
 EQ("init.analog_preserve_other_pins",ANSB,0x05|(1u<<EXPECT_PIN));
 EQ("init.input_preserve_other_pins",OEB,0xFF&~(1u<<EXPECT_PIN));
 EQ("init.pullup_preserve_other_pins",PUB,0xFF&~(1u<<EXPECT_PIN));
 EQ("init.pulldown_preserve_other_pins",PDB,0xFF&~(1u<<EXPECT_PIN));
 EQ("init.GIEH_preserved",GIEH,1); EQ("init.GIEL_preserved",GIEL,1);
 EQ("init.timer_config_preserved",T1CR,12); EQ("init.timer_enable_preserved",TC1EN,1);
 EQ("idle.no_result",Demo_Adc0_Read(&value),0); EQ("idle.output_preserved",value,0xBEEF);
 for(i=0;i<sizeof(samples)/sizeof(samples[0]);i++) {
   EQ("start.accepted",Demo_Adc0_Start(),1); EQ("start.hardware_started",ADS,1);
   for(j=0;j<1000;j++) {
     Demo_Adc0_Poll(); EQ("busy.reject_start",Demo_Adc0_Start(),0);
   }
   EQ("busy.hardware_flag_preserved",ADS,1);
   EQ("busy.no_premature_result",Demo_Adc0_Read(&value),0);
   EQ("busy.output_preserved",value,0xBEEF);
   AD=(unsigned short)samples[i]; ADS=0; Demo_Adc0_Poll();
   EQ("ready.no_auto_start",ADS,0); EQ("ready.reject_start",Demo_Adc0_Start(),0);
   EQ("ready.null_output_does_not_consume",Demo_Adc0_Read(0),0);
   EQ("ready.read",Demo_Adc0_Read(&value),1);
   EQ("ready.sample",value,samples[i]&0x0FFFu);
   value=0xBEEF;
   EQ("consumed.read_once",Demo_Adc0_Read(&value),0);
   EQ("consumed.output_preserved",value,0xBEEF);
 }
 puts("same API / nonblocking / busy rejection / sample / consume-once contract passed");
 return 0;
}
'''

INTEGRATION = r'''
#include <stdio.h>
#include <setjmp.h>
#include "COMMON.H"
#include "modules/demo_adc0.h"
void firmware_main(void); void int_h(void);
extern volatile unsigned int app_adc_value;
static jmp_buf done;
static unsigned int loops,polls,ticks;
void host_watchdog(void) {
 if(++loops>40) longjmp(done,1);
 TC1IF=1; int_h();
 if(ADS) { AD=0x345; ADS=0; }
}
void observed_poll(void) { polls++; Demo_Adc0_Poll(); }
void observed_tick(void) { ticks++; TC1_Work(); }
int main(void) {
 if(setjmp(done)==0) firmware_main();
 if(polls!=40 || ticks!=40 || b_TC1_Flag!=0 || TC1IF!=0 || IOB1!=0 ||
    TC1PRH!=3 || TC1PRL!=0xE7 || app_adc_value!=0x345) {
   fprintf(stderr,"integration: polls=%u ticks=%u flag=%u IRQ=%u pin=%u period=%u value=%u\n",
           polls,ticks,b_TC1_Flag,TC1IF,IOB1,(TC1PRH<<8)|TC1PRL,app_adc_value);
   return 1;
 }
 puts("target main sampling and original TC1 interrupt/heartbeat passed"); return 0;
}
'''


def check(workspace: Path, out: Path, seed: Path = SEED) -> dict:
    workspace, out, seed = workspace.resolve(), out.resolve(), seed.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []

    def require(name, passed, detail):
        results.append({"id": name, "passed": bool(passed), "detail": detail})

    for path in (seed / "source").rglob("*"):
        if path.is_file():
            current = workspace / path.relative_to(seed)
            require("source_preserved:" + path.relative_to(seed).as_posix(),
                    current.is_file() and current.read_bytes() == path.read_bytes(),
                    "source module and provenance must remain unchanged")
    target = workspace / "target"
    for name in ("COMMON.H", "MASSESCHIP_DEFINE.H", "TC1_Work.c"):
        path = target / name
        require("target_preserved:" + name,
                path.is_file() and path.read_bytes() == (seed / "target" / name).read_bytes(),
                "original target vendor file must remain unchanged")
    try:
        require("target_system_init", function_body(target / "M9F6820_TC1.C", "SYSTEM_INIT")
                == function_body(seed / "target/M9F6820_TC1.C", "SYSTEM_INIT"),
                "preserve target clock and global IO initialization")
        old, new = configparser.ConfigParser(), configparser.ConfigParser()
        old.read_string(text(seed / "target/M9F6820_TC1.prj"))
        new.read_string(text(target / "M9F6820_TC1.prj"))
        require("target_project_settings", dict(old["Settings"]) == dict(new["Settings"]),
                "preserve target device and options")
        files = [value.replace("\\", "/") for value in new["Project Files"].values()]
        required = {"modules/demo_adc0.c", "modules/demo_adc0.h"}
        original = {value.replace("\\", "/") for value in old["Project Files"].values()}
        require("target_project_files", original | required <= set(files)
                and new.getint("File Count", "Count") == len(files)
                and all((target / name).is_file() for name in files),
                "keep original files; register modules/demo_adc0.c and .h; match file count")
    except (OSError, KeyError, ValueError, StopIteration, configparser.Error) as exc:
        require("target_project_structure", False, str(exc))
        files = []
    require("migration_notes", (target / "MIGRATION.md").is_file(),
            "record retained behavior, hardware differences, sources and checks")

    compiler = os.environ.get("CC", "gcc")
    env = {**os.environ, "PATH": str(Path(compiler).resolve().parent)
           + os.pathsep + os.environ.get("PATH", "")}
    header, registers = host_files()

    def run(name: str, command: list[str], timeout: float = 25) -> bool:
        try:
            process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                     errors="replace", timeout=timeout, env=env)
            require(name, process.returncode == 0,
                    {"exit_code": process.returncode, "stdout": process.stdout,
                     "stderr": process.stderr})
            return process.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            require(name, False, str(exc))
            return False

    for name, folder, relative, channel, divider, pin in (
        ("source", workspace / "source", "agent_demo/modules/demo_adc0", 15, 7, 7),
        ("target", target, "modules/demo_adc0", 12, 2, 4),
    ):
        build = out / name
        build.mkdir(exist_ok=True)
        (build / "zc.h").write_text(header, encoding="utf-8")
        (build / "registers.c").write_text(registers, encoding="utf-8")
        (build / "module.h").write_text(
            f'#include "{(folder / relative).as_posix()}.h"\n', encoding="utf-8",
        )
        (build / "behavior.c").write_text(BEHAVIOR, encoding="utf-8")
        common = [compiler, "-std=c11", "-fcommon", "-Wall", "-Werror", "-Wno-main",
                  "-I", str(build), "-I", str(folder)]
        executable = build / "behavior.exe"
        if run(name + ":compile_behavior", [*common, f"-DEXPECT_CHANNEL={channel}",
               f"-DEXPECT_DIVIDER={divider}", f"-DEXPECT_PIN={pin}", "-x", "c",
               str(folder / (relative + ".c")), str(build / "registers.c"),
               str(build / "behavior.c"), "-o", str(executable)]):
            run(name + ":shared_behavior", [str(executable)], timeout=4)
        if name == "target" and files:
            main = target / "M9F6820_TC1.C"
            (build / "integration.c").write_text(INTEGRATION, encoding="utf-8")
            obj = build / "main.o"
            if run("target:compile_main", [*common, "-Dmain=firmware_main",
                   "-DDemo_Adc0_Poll=observed_poll", "-DTC1_Work=observed_tick", "-x", "c",
                   "-c", str(main), "-o", str(obj)]):
                executable = build / "integration.exe"
                sources = [str(target / path) for path in files
                           if Path(path).suffix.lower() == ".c" and path != main.name]
                if run("target:link_project", [*common, str(obj), "-x", "c", *sources,
                       str(build / "registers.c"), str(build / "integration.c"),
                       "-o", str(executable)]):
                    run("target:old_behavior_and_sampling", [str(executable)], timeout=4)
    return {"passed": all(row["passed"] for row in results), "checks": results,
            "validation_layer": "host_module_behavior", "vendor_compile": "not_run",
            "hardware": "not_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.workspace, args.out)
    (args.out / "migration_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    # Return direct expected/actual diagnostics; the detailed report stays alongside builds.
    print(json.dumps({"passed": report["passed"],
                      "failures": [row for row in report["checks"] if not row["passed"]]},
                     ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
