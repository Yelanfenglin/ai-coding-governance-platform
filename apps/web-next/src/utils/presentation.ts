import { parsePatch } from "diff";
import type { TaskEvent } from "../stores/task";

export interface ToolActivity {
  id: string;
  name: string;
  arguments: unknown;
  content: string;
  status: "running" | "completed" | "failed" | "interrupted";
}

export interface TraceStep {
  id: string;
  step: number;
  reasoning: string;
  assistantFinished: boolean;
  finished: boolean;
  events: TaskEvent[];
}

export function traceSteps(events: TaskEvent[]): TraceStep[] {
  const steps = new Map<string, TraceStep>();
  for (const event of events) {
    if (
      typeof event.payload.step !== "number" ||
      typeof event.payload.turn !== "number"
    )
      continue;
    const id = `${event.payload.turn}:${event.payload.step}`;
    let row = steps.get(id);
    if (!row) {
      row = {
        id,
        step: event.payload.step,
        reasoning: "",
        assistantFinished: false,
        finished: false,
        events: [],
      };
      steps.set(id, row);
    }
    if (event.type === "AssistantReasoningDelta")
      row.reasoning += String(event.payload.text ?? "");
    if (event.type === "ProviderRetryStarted") row.reasoning = "";
    if (event.type === "AssistantTurnEvent") {
      const message = objectValue(event.payload.message);
      if (typeof message.reasoning_content === "string")
        row.reasoning = message.reasoning_content;
      row.assistantFinished = true;
    }
    if (event.type === "StepFinished") row.finished = true;
    if (event.type === "ToolStarted" || event.type === "ToolFinished")
      row.events.push(event);
  }
  return [...steps.values()].filter(
    (row) => row.reasoning || row.events.length,
  );
}

// Cline's ToolGroupRenderer informs the grouped disclosure; pairing uses our runtime call ids.
export function toolActivities(
  events: TaskEvent[],
  active: boolean,
): ToolActivity[] {
  const calls = new Map<string, ToolActivity>();
  for (const event of events) {
    if (event.type !== "ToolStarted" && event.type !== "ToolFinished") continue;
    const call = event.payload.call as {
      id: string;
      name: string;
      arguments: unknown;
    };
    if (event.type === "ToolStarted") {
      calls.set(call.id, {
        ...call,
        content: "",
        status: active ? "running" : "interrupted",
      });
    } else {
      const result = event.payload.result as {
        content: string;
        is_error: boolean;
      };
      const row = calls.get(call.id);
      if (row)
        Object.assign(row, {
          content: result.content,
          status: result.is_error ? "failed" : "completed",
        });
    }
  }
  return [...calls.values()];
}

const labels: Record<string, [string, string]> = {
  list_files: ["浏览工程", "folder"],
  read_file: ["读取文件", "file"],
  find_symbols: ["定位源码符号", "code"],
  find_symbol_references: ["查找符号引用", "search"],
  write_file: ["写入文件", "code"],
  edit_file: ["修改文件", "code"],
  run_command: ["运行命令", "terminal"],
  search_spec: ["检索规格书", "search"],
  read_context: ["展开规格依据", "book"],
  read_skill: ["读取开发技能", "book"],
  verify_project: ["检查工程", "check"],
  record_implementation: ["记录实现依据", "book"],
  check_project_manifest: ["核对工程清单", "check"],
  list_official_examples: ["查找官方例程", "folder"],
  read_official_example: ["读取官方例程", "book"],
  remember_fact: ["保存确认事实", "book"],
  replace_fact: ["更新事实", "book"],
  forget_fact: ["移除记忆", "book"],
  search_sessions: ["查找历史会话", "search"],
};
export const toolLabel = (name: string) => labels[name]?.[0] ?? name;
export const toolIcon = (name: string) => labels[name]?.[1] ?? "code";

export function objectValue(value: unknown): Record<string, unknown> {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return {};
    }
  }
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function toolSubject(tool: ToolActivity): string {
  const args = objectValue(tool.arguments);
  if (Array.isArray(args.command)) return args.command.join(" ");
  return String(args.path ?? args.question ?? args.key ?? args.module ?? "");
}

export function readableOutput(content: string): string {
  const value = objectValue(content);
  return typeof value.output === "string"
    ? value.output
    : typeof value.content === "string"
      ? value.content
      : typeof value.diff === "string"
        ? value.diff
        : content;
}

export interface Evidence {
  document: string;
  chunk: string;
  product: string;
  version: string;
  pages: string;
  section: string;
  text: string;
}

export function evidenceItems(
  rows: Array<Record<string, unknown>>,
): Evidence[] {
  const items = new Map<string, Evidence>();
  for (const row of rows) {
    if (row.is_error) continue;
    const content = String(row.content ?? "");
    const headers = [
      ...content.matchAll(
        /^(?:\[\d+\] )?document=(\S+) chunk=(\S+)([^\n]*)\n?/gm,
      ),
    ];
    for (let index = 0; index < headers.length; index++) {
      const match = headers[index]!;
      const fields = match[3]!;
      const field = (name: string) =>
        fields.match(new RegExp(`(?:^|\\s)${name}=([^\\s]+)`))?.[1] ?? "";
      const item = {
        document: match[1]!,
        chunk: match[2]!,
        product: field("product"),
        version: field("version"),
        pages: field("pages").replace(/^(\d+)-\1$/, "$1"),
        section: fields.match(/section=(.*?)(?: kind=| score=|$)/)?.[1] ?? "",
        text: content
          .slice(match.index! + match[0].length, headers[index + 1]?.index)
          .trim(),
      };
      const key = `${item.document}:${item.chunk}`;
      if (!items.has(key) || item.text.length > items.get(key)!.text.length)
        items.set(key, item);
    }
  }
  return [...items.values()];
}

export interface DiffLine {
  kind: string;
  text: string;
  old: number | null;
  next: number | null;
}
export function fileDiff(patch: string, file: string): DiffLine[] {
  const part = parsePatch(patch).find(
    (item) =>
      item.newFileName === `after/${file}` ||
      item.oldFileName === `before/${file}`,
  );
  return (part?.hunks ?? []).flatMap((hunk) => {
    let old = hunk.oldStart;
    let next = hunk.newStart;
    const header: DiffLine = {
      kind: "hunk",
      text: `@@ -${hunk.oldStart},${hunk.oldLines} +${hunk.newStart},${hunk.newLines} @@`,
      old: null,
      next: null,
    };
    return [
      header,
      ...hunk.lines.map((text) => {
        const kind =
          text[0] === "+" ? "added" : text[0] === "-" ? "removed" : "context";
        return {
          kind,
          text,
          old: text[0] === "+" || text[0] === "\\" ? null : old++,
          next: text[0] === "-" || text[0] === "\\" ? null : next++,
        };
      }),
    ];
  });
}

export const verificationLabel = (status?: string) =>
  status === "passed"
    ? "配置的检查已通过"
    : status === "failed"
      ? "检查未通过"
      : "未配置独立验收";
