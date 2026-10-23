"""最小 CLI：一个问题 → 一次 run → 打印回答。

环境变量：

- LLM：`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`
- RAG：`RAG_API_URL`（默认 `http://127.0.0.1:8000`）/ `RAG_API_KEY`

不写死的策略：模型和 RAG 地址都来自环境或命令行，CLI 只负责把它们接起来。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .compaction import CompactionPolicy, ModelSummarizer, leading_system_count
from .contracts import (
    AssistantTextDelta,
    AssistantTurnEvent,
    Budget,
    CompactionWarning,
    ContextCompacted,
    MemoryRecalled,
    Message,
    MessageEntry,
    ToolFinished,
    ToolStarted,
)
from .delivery import DeliveryRecorder
from .delivery import project_files as _project_files
from .demos import DemoProject
from .demos.catalog import PRODUCT, SPEC_VERSION
from .demos.firmware import FirmwareProject
from .demos.manifest_check import STATE_FILE as MANIFEST_STATE_FILE
from .demos.manifest_check import check_demo_manifest
from .demos.tools import demo_tools, skill_prompt
from .goal_driver import GoalRoundDriver
from .goal_tools import goal_tools, todo_tool
from .goals import GoalService
from .inbox import Inbox
from .loop import RunOptions, run_agent, system_message, user_message
from .memory import DEFAULT_MEMORY_CHARS, DEFAULT_USER_CHARS, JsonMemoryStore
from .memory_tools import memory_tools, search_sessions_tool
from .project_verification import ProjectVerification
from .prompts import react_system_prompt
from .providers.openai_compatible import OpenAiCompatibleProvider
from .rag_tools import RagClient, RagClientConfig, rag_tools
from .session import JsonlSessionStore, repair_tool_pairing
from .session_search import SessionSearch
from .subagents import SubagentRunner
from .tools import ToolRegistry
from .workspace_tools import example_tools, workspace_prompt, workspace_tools

EXIT_OK = 0
EXIT_FAILED = 1
#: 预算用尽也是非零退出，但和失败区分开，便于脚本判断是"没跑完"还是"跑错了"。
EXIT_BUDGET = 3
#: 模型认为完成了，但工程清单与磁盘代码不自洽：交付不成立，单独给一个码。
EXIT_MANIFEST = 4
EXIT_VERIFICATION = 5
DEFAULT_CONTEXT_WINDOW = 1_000_000


def _workspace_target(workspace: str) -> Path:
    selected = Path(workspace).resolve()
    return selected.parent if selected.suffix.lower() == ".prj" else selected


def _manifest_report(target: Path) -> dict | None:
    """工程带清单才核对；不带清单的普通目录不参与。"""

    if not (target / MANIFEST_STATE_FILE).is_file():
        return None
    return check_demo_manifest(target)


def _final_answer(messages: list[Message], start: int = 0) -> str:
    """只取 assistant 输出；没有输出就是空串（不能拿用户问题或工具结果充数）。

    `start` 是本轮新增消息的起点：续接会话时历史里已有旧的 assistant 回答，
    本轮若在生成前失败，不能把上一轮的答案当成本轮答案。
    """
    for message in reversed(messages[start:]):
        if message.role == "assistant":
            return message.content
    return ""


#: 历史里缺失的工具结果补这条说明（**不重跑工具**：副作用可能已经发生过）。
INTERRUPTED_TOOL_RESULT = (
    "Error: tool call was interrupted before it returned a result"
)


def _repair_pairing(messages: list[Message]) -> list[Message]:
    return repair_tool_pairing(messages, INTERRUPTED_TOOL_RESULT)


def _continuation(
    history: list[Message], question: str, extra_prompt: str = "",
) -> list[Message]:
    """把本次问题接到已有上下文后面（上游交互客户端从 session 接续）。

    规则：沿用会话**自己的** system head，不重新拼一份 —— 否则同一会话里会出现两条
    system，旧的固定前缀被替换，provider 的前缀缓存和已确立的指令都会变。
    历史为空时（新会话 / `--no-history`）才用当前 system prompt。
    接之前先补齐工具调用配对：中断留下的半截历史不能原样送下一轮。
    """
    prompt = react_system_prompt() + (f"\n\n{extra_prompt}" if extra_prompt else "")
    if not history:
        return [system_message(prompt), user_message(question)]
    history = _repair_pairing(history)
    prefix = leading_system_count(history)
    head = history[:prefix] if prefix else [system_message(prompt)]
    return [*head, *history[prefix:], user_message(question)]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-runtime", description="Run one question through the agent runtime."
    )
    parser.add_argument("question", nargs="?", help="question to ask")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-flash"))
    parser.add_argument(
        "--thinking", choices=("enabled", "disabled"),
        default=os.environ.get("OPENAI_THINKING", "enabled"),
    )
    parser.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"),
        default=os.environ.get("OPENAI_REASONING_EFFORT", "high"),
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--product", default=None, help="default product filter for search_spec")
    parser.add_argument("--version", default=None, help="default version filter for search_spec")
    parser.add_argument(
        "--rag-url", default=os.environ.get("RAG_API_URL", "http://127.0.0.1:8000/api/rag")
    )
    parser.add_argument("--rag-key", default=os.environ.get("RAG_API_KEY", ""))
    parser.add_argument(
        "--session",
        default=None,
        help="resume this JSONL session (created when missing) and append the run to it",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="append to --session without loading its earlier messages",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps")
    parser.add_argument(
        "--max-goal-rounds", type=int, default=256,
        help="default round cap for new goals; create_goal may override it (default 256)",
    )
    parser.add_argument(
        "--resume-goal", action="store_true",
        help="explicit host-authorized goal resume, including a paused goal",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="stop after this many seconds"
    )
    parser.add_argument("--json", action="store_true", help="print a JSON result instead of text")
    parser.add_argument(
        "--delivery-dir", type=Path,
        help="write delivery.json, changes.diff and a review report",
    )
    parser.add_argument(
        "--verification", type=Path,
        help="explicit host JSON with command (argv), cwd and timeout; enables verify_project",
    )
    parser.add_argument(
        "--state-dir", default=os.environ.get("AGENT_STATE_DIR"),
        help="memory/index directory; defaults to .agent-runtime inside the selected workspace",
    )
    parser.add_argument("--no-memory", action="store_true", help="disable durable memory tools")
    parser.add_argument("--memory-chars", type=int, default=DEFAULT_MEMORY_CHARS)
    parser.add_argument("--profile-chars", type=int, default=DEFAULT_USER_CHARS)
    parser.add_argument(
        "--context-window", type=int,
        default=int(os.environ.get("AGENT_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))),
        help="local context pressure budget; set for the model in use (default 1000000)",
    )
    parser.add_argument("--no-compaction", action="store_true", help="disable automatic compaction")
    parser.add_argument(
        "--subagents", action="store_true",
        help="enable foreground read-only subagents with a shared task budget",
    )
    project_group = parser.add_mutually_exclusive_group()
    project_group.add_argument(
        "--workspace", default=None,
        help="enable local programming tools in any project directory",
    )
    project_group.add_argument(
        "--demo-project", default=None, help="enable module generation in this project",
    )
    project_group.add_argument(
        "--firmware-project", default=None,
        help="integrate modules into an existing M9P6362 ZCW-ICE project directory or .prj",
    )
    parser.add_argument("--examples-root", default=os.environ.get("DEMO_EXAMPLES_ROOT"))
    return parser


async def run_once(
    args: argparse.Namespace, *,
    event_sink: Callable[[object], None] | None = None,
    result_sink: Callable[[dict, int], None] | None = None,
    cancel_event: asyncio.Event | None = None,
    console: bool = True,
) -> int:
    if not args.question:
        print("a question is required", file=sys.stderr)
        return EXIT_FAILED
    if not args.base_url:
        print("OPENAI_BASE_URL (or --base-url) is required", file=sys.stderr)
        return EXIT_FAILED
    if (args.context_window <= 0 or args.memory_chars <= 0 or args.profile_chars <= 0
            or args.max_goal_rounds <= 0):
        print("context and memory budgets must be positive", file=sys.stderr)
        return EXIT_FAILED
    module_project = args.demo_project or args.firmware_project
    if module_project:
        if not args.examples_root:
            print(
                "module generation requires --examples-root / DEMO_EXAMPLES_ROOT", file=sys.stderr,
            )
            return EXIT_FAILED
        if args.product not in (None, PRODUCT) or args.version not in (None, SPEC_VERSION):
            print(f"demo generation currently supports {PRODUCT} {SPEC_VERSION}", file=sys.stderr)
            return EXIT_FAILED
        args.product, args.version = PRODUCT, SPEC_VERSION

    def make_provider():
        return OpenAiCompatibleProvider(
            base_url=args.base_url, model=args.model, api_key=args.api_key or None,
            thinking=args.thinking, reasoning_effort=args.reasoning_effort,
        )

    provider = make_provider()
    client = RagClient(RagClientConfig(base_url=args.rag_url, api_key=args.rag_key or None))

    registry = ToolRegistry()
    for tool in rag_tools(client, default_product=args.product, default_version=args.version):
        registry.register(tool)
    if module_project:
        project_type = FirmwareProject if args.firmware_project else DemoProject
        for tool in demo_tools(project_type(module_project, args.examples_root)):
            registry.register(tool)
    workspace = args.workspace or module_project
    selected: Path | None = None
    if workspace:
        selected = _workspace_target(workspace)
        for tool in workspace_tools(selected, manifest=True if module_project else None):
            registry.register(tool)
        if args.examples_root:
            for tool in example_tools(args.examples_root):
                registry.register(tool)

    # Project state remains workspace-scoped. Delegation needs a parent log for
    # completed-turn fork boundaries even when --session was not supplied.
    state_dir = (
        Path(args.state_dir) if args.state_dir else (selected or Path.cwd()) / ".agent-runtime"
    )
    verification = None
    if args.verification:
        if selected is None:
            print("--verification requires a selected project", file=sys.stderr)
            return EXIT_FAILED
        if args.delivery_dir is None:
            args.delivery_dir = state_dir / "deliveries" / f"{datetime.now(UTC):%Y%m%dT%H%M%S%f}"
        try:
            verification = ProjectVerification(
                args.verification, selected,
                exclude=(state_dir.resolve(), args.delivery_dir.resolve()),
            )
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_FAILED
        registry.register(verification.tool())
    session_path = args.session
    if session_path is None and args.subagents:
        session_path = state_dir / "sessions" / f"{datetime.now(UTC):%Y%m%dT%H%M%S%f}.jsonl"
    session = JsonlSessionStore(session_path) if session_path else None
    goals = GoalService(session, default_max_goal_rounds=args.max_goal_rounds)
    if session is not None:
        goals.replay(await session.load_events())
    goal_driver = GoalRoundDriver(goals)
    if args.resume_goal:
        current_goal = goals.get()
        if current_goal is None:
            print("no current goal to resume", file=sys.stderr)
            return EXIT_FAILED
        await goals.update(
            {"id": current_goal["id"], "revision": current_goal["revision"]}, "resume",
        )
    for tool in [*goal_tools(goal_driver), todo_tool(goal_driver, allow_parallel=True)]:
        registry.register(tool)
    started: list[str] = []
    turn_messages: list[Message] = []
    compactions: list[dict[str, int | None]] = []
    compaction_warnings: list[str] = []
    memory_recalled = 0
    applies_to = {
        key: value for key, value in {"product": args.product, "version": args.version}.items()
        if value is not None
    }
    facts = profile = None
    search = None
    if not args.no_memory:
        facts = JsonMemoryStore(state_dir / "facts.jsonl", max_chars=args.memory_chars)
        profile = JsonMemoryStore(
            state_dir / "profile.jsonl", max_chars=args.profile_chars, default_scope="user",
        )
        for tool in memory_tools(
            facts, user_profile=profile, applies_to=applies_to,
            default_source=f"session:{session.path.stem}" if session else "user conversation",
        ):
            registry.register(tool)
        search = SessionSearch(state_dir / "history.sqlite")
        if session is not None:
            await search.index_file(session.path, session=str(session.path.resolve()))
        registry.register(search_sessions_tool(search))

    # 续接：先把历史读出来，再落本轮问题 —— 顺序反过来会把当前问题算进历史。
    history: list[Message] = []
    if session is not None and not args.no_history:
        history = await session.load_messages()
    delivery = (
        DeliveryRecorder(selected, args.delivery_dir, state_dir=state_dir)
        if args.delivery_dir else None
    )
    extra_prompt = workspace_prompt(selected) if workspace else ""
    if delivery is not None and delivery.implementation is not None:
        registry.register(delivery.implementation.tool())
    if module_project:
        extra_prompt += "\n\n" + skill_prompt()
    subagents = None
    parent_inbox = Inbox()
    if args.subagents:
        child_tools = ToolRegistry()
        for tool in rag_tools(client, default_product=args.product, default_version=args.version):
            child_tools.register(tool)
        if selected:
            for tool in workspace_tools(selected):
                if tool.name in {"list_files", "read_file", "check_project_manifest"}:
                    child_tools.register(tool)
        if args.examples_root:
            for tool in example_tools(args.examples_root):
                child_tools.register(tool)
        if search is not None:
            child_tools.register(search_sessions_tool(search))

        def child_options():
            child_provider = make_provider()
            return RunOptions(
                provider=child_provider, tools=child_tools,
                messages=[system_message(
                    "You are a research/review subagent. Use the supplied read-only tools. "
                    "Return concise findings with file/line or document/chunk/page references. "
                    "State uncertainty and checks not performed; the parent integrates changes.\n"
                    "Leave a model response in the shared budget for the parent to report.\n"
                    f"Workspace: {selected}; product: {args.product}; version: {args.version}",
                )],
                memory=facts, user_profile=profile, applies_to=applies_to,
                compaction=(None if args.no_compaction else CompactionPolicy(
                    context_window=args.context_window, summarizer=ModelSummarizer(child_provider),
                )),
            )

        subagents = SubagentRunner(
            child_options, state_dir / "subagents", parent_session=session,
            parent_inbox=parent_inbox,
        )
        for tool in subagents.tools():
            registry.register(tool)
        extra_prompt += (
            "\n\nUse subagent only when focused delegation is worthwhile. Choose spawn or fork, "
            "foreground or background, and whether follow-ups need a continuable session. "
            "Children have read-only research/review tools; give current constraints and evidence "
            "references. wait_agent collects results; send_message steers a running child or "
            "resumes an idle continuable child. Their steps/time share the task budget; you own "
            "integration and final validation. Collect required results before the final answer: "
            "this CLI closes unfinished children on return, it is not a background daemon."
        )
    input_question = args.question
    if delivery is not None and delivery.implementation is not None:
        # Per-turn guidance also reaches sessions created before this tool existed.
        input_question += (
            "\n\n[实现依据] 工程修改并检查后，用 record_implementation 批量记录关键实现："
            "需求、配置理由、实际源码行、已取得的规格引用和检查调用。解释由你提供，"
            "宿主核对定位与执行记录；未验证事项明确列出。只记录本轮实际工作，不为凑记录"
            "重复检索或检查。没有工程修改时按需使用。"
            "规格明确给出的事实与基于前提的计算、推导分开表述，不把推导写成手册明文。"
            "最终答复按本轮工具输出说明实际检查：主机编译、行为模拟、厂商编译和板卡实测"
            "分别描述，不沿用旧清单的验证级别，也不遗漏已执行的检查。"
        )
    if verification is not None:
        input_question += (
            "\n\n[宿主验收] 本轮已配置 verify_project，修改后调用它。失败时根据实际输出修复"
            "并复验。它的命令与工作目录见工具定义；不修改验收判据来获得通过。"
            "交付时宿主还会检查最终版本，模型的完成声明不能代替验收结果。"
        )
    messages = _continuation(history, input_question, extra_prompt)
    context_anchor = (
        await session.load_context_anchor() if session is not None and history else None
    )

    # 本轮输入要落盘：否则重放会话时看不到用户问了什么。
    if session is not None:
        if not await session.load_nodes():
            # 固定 system head 也要可重放；新会话持久化一次，记忆快照不落盘。
            for head in messages[: leading_system_count(messages)]:
                await session.append(MessageEntry(0, 0, head, _now()))
        await session.append(MessageEntry(0, 0, user_message(input_question), _now()))

    def on_event(event: object) -> None:
        nonlocal memory_recalled
        if delivery is not None:
            delivery.observe(event)
        if event_sink is not None:
            event_sink(event)
        if isinstance(event, AssistantTextDelta):
            if console and not args.json:
                sys.stdout.write(event.text)
                sys.stdout.flush()
        elif isinstance(event, AssistantTurnEvent):
            turn_messages.append(event.message)
        elif isinstance(event, MemoryRecalled):
            memory_recalled += len(event.items)
        elif isinstance(event, ContextCompacted):
            compactions.append({
                "before": event.context_tokens, "after": event.context_tokens_after,
            })
        elif isinstance(event, CompactionWarning):
            compaction_warnings.append(event.message)
            print(f"[compaction warning] {event.message}", file=sys.stderr)
        elif isinstance(event, ToolStarted):
            started.append(event.call.name)
        elif isinstance(event, ToolFinished) and args.json and event.result.is_error:
            print(f"[tool error] {event.call.name}: {event.result.content}", file=sys.stderr)

    before_delivery = (
        _project_files(selected)
        if selected and (module_project or (selected / MANIFEST_STATE_FILE).is_file())
        else None
    )
    try:
        result = await run_agent(RunOptions(
            provider=provider,
            tools=registry,
            messages=messages,
            session=session,
            memory=facts, user_profile=profile, applies_to=applies_to,
            context_anchor=context_anchor,
            goal_driver=goal_driver,
            inbox=parent_inbox,
            compaction=(
                None if args.no_compaction else CompactionPolicy(
                    context_window=args.context_window, summarizer=ModelSummarizer(provider),
                )
            ),
            budget=(
                Budget(max_steps=args.max_steps, max_seconds=args.max_seconds)
                if args.max_steps is not None or args.max_seconds is not None
                else None
            ),
            cancel_event=cancel_event,
            on_event=on_event,
        ))
    finally:
        if subagents is not None:
            await subagents.close()
        if search is not None and session is not None:
            await search.index_file(session.path, session=str(session.path.resolve()))

    # A completed model turn does not certify the final source. No extra LLM/retry loop.
    if verification is not None and result.status == "completed":
        await verification.run(phase="delivery")
    acceptance = (verification.report() if verification is not None else {
        "status": "unverified", "reason": "no host verification configured", "runs": [],
    })
    if acceptance["status"] == "failed":
        print("[verification failed] " + json.dumps(acceptance["runs"][-1], ensure_ascii=False),
              file=sys.stderr)
    elif verification is not None and not args.json:
        print(f"\n[verification] {acceptance['status']}", file=sys.stderr)

    # 始终保留清单诊断；硬门禁只约束本轮实际改动后的工程交付。
    manifest = await asyncio.to_thread(_manifest_report, selected) if selected else None
    delivery_changed = (
        selected is not None and manifest is not None
        and (before_delivery is None or before_delivery != _project_files(selected))
    )
    if manifest is not None and not manifest["ok"]:
        print(
            "[manifest] 工程清单与代码不一致：" + "；".join(
                f"{name}" for name in manifest["failed"]
            ),
            file=sys.stderr,
        )
        for item in manifest["items"]:
            if not item["ok"]:
                print(f"  {item['check']}: {item['detail']}", file=sys.stderr)

    if result.status == "completed":
        exit_code = (
            EXIT_MANIFEST if delivery_changed and manifest is not None and not manifest["ok"]
            else EXIT_OK
        )
    elif result.status == "budget_exceeded":
        exit_code = EXIT_BUDGET
    else:
        exit_code = EXIT_FAILED
    if exit_code == EXIT_OK and acceptance["status"] == "failed":
        exit_code = EXIT_VERIFICATION

    response = {
        "status": result.status, "turns": result.turns, "steps": result.steps,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "tool_calls": started, "answer": _final_answer(turn_messages),
        "memory_recalled": memory_recalled, "goal": goals.get(), "todos": goal_driver.todos,
        "compactions": compactions, "compaction_warnings": compaction_warnings,
        "resumed_messages": len(history), "manifest": manifest, "error": result.error,
        "verification": acceptance,
    }
    if subagents is not None:
        response["subagents"] = subagents.records
        response["total_steps"] = result.total_steps
    if delivery is not None:
        path = delivery.finish(
            result, question=args.question, response=response, exit_code=exit_code,
            settings={
                "model": args.model, "thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort,
                "max_steps": args.max_steps, "max_seconds": args.max_seconds,
                "context_window": args.context_window,
                "memory_enabled": not args.no_memory,
                "compaction_enabled": not args.no_compaction,
                "product": args.product, "version": args.version,
                "subagents_enabled": args.subagents,
            },
            session=session.path if session else None,
        )
        response["delivery"] = str(path)

    if result_sink is not None:
        result_sink(response, exit_code)
    if console and args.json:
        print(
            json.dumps(
                response,
                ensure_ascii=False,
                indent=2,
            )
        )
    elif console and result.status != "completed":
        print(f"\n[{result.status}] {result.error or ''}", file=sys.stderr)

    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run_once(args))


if __name__ == "__main__":
    raise SystemExit(main())
