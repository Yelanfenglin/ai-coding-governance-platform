"""不需要模型的本地 Demo 入口：与 Agent 工具复用同一生成函数。"""

import argparse
import json
import os

from .demos import DemoError, DemoProject
from .demos.firmware import FirmwareProject


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a module from reviewed official examples",
    )
    parser.add_argument("command", choices=["catalog", "show", "generate"])
    parser.add_argument("--examples-root", default=os.environ.get("DEMO_EXAMPLES_ROOT"))
    parser.add_argument("--project", default="runs/demo-project")
    parser.add_argument("--module", choices=["pwm0", "timer0", "timer1"])
    parser.add_argument("--parameters", default="{}", help="JSON configuration")
    parser.add_argument("--reserved-resources", default=None, help="comma separated resource names")
    parser.add_argument("--existing", action="store_true", help="use an existing ZCW-ICE project")
    parser.add_argument("--preview", action="store_true", help="preview existing-project changes")
    args = parser.parse_args(argv)
    if not args.examples_root:
        parser.error("--examples-root or DEMO_EXAMPLES_ROOT is required")
    project_type = FirmwareProject if args.existing else DemoProject
    project = project_type(args.project, args.examples_root)
    try:
        if args.command == "catalog":
            result = project.catalog()
        elif args.command == "show":
            result = project.read()
        elif args.command == "generate":
            if not args.module:
                parser.error("generate requires --module")
            result = project.generate(
                args.module,
                json.loads(args.parameters),
                reserved_resources=(
                    [value.strip() for value in args.reserved_resources.split(",") if value.strip()]
                    if args.reserved_resources is not None
                    else None
                ),
                **({"preview": args.preview} if args.existing else {}),
            )
    except (DemoError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
