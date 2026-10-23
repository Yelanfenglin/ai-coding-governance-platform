import { describe, expect, it } from "vitest";
import { renderMarkdown } from "./markdown";
import {
  evidenceItems,
  fileDiff,
  toolActivities,
  traceSteps,
} from "./presentation";
import type { TaskEvent } from "../stores/task";

describe("Agent message rendering", () => {
  it("renders technical Markdown with highlighted code, tables and safe external links", () => {
    const html = renderMarkdown(
      "# ADC 配置\n\n```c\nint value = 2;\n```\n\n| 型号 | 版本 |\n| --- | --- |\n| M9F6820 | V1.7 |\n\n[依据](https://example.com/manual)",
    );
    expect(html).toContain("<h1>ADC 配置</h1>");
    expect(html).toContain("language-c");
    expect(html).toContain("hljs-number");
    expect(html).toContain("<td>M9F6820</td>");
    expect(html).toContain('rel="noopener noreferrer"');
  });

  it("does not execute raw model HTML or javascript links", () => {
    const html = renderMarkdown(
      "<img src=x onerror=alert(1)>\n\n<script>alert(1)</script>\n\n[bad](javascript:alert(1))\n\n```unknown\n<script>alert(1)</script>\n```",
    );
    expect(html).not.toMatch(/<script|<img|href="javascript:/);
    expect(html).toContain("&lt;script&gt;");
  });

  it("pairs concurrent tool results by call id and retains malformed raw arguments", () => {
    const a = { id: "a", name: "read_file", arguments: "{broken" };
    const b = {
      id: "b",
      name: "run_command",
      arguments: { command: ["python", "-V"] },
    };
    const events: TaskEvent[] = [
      { sequence: 1, type: "ToolStarted", payload: { call: a } },
      { sequence: 2, type: "ToolStarted", payload: { call: b } },
      {
        sequence: 3,
        type: "ToolFinished",
        payload: {
          call: b,
          result: { content: "Python 3.13", is_error: false },
        },
      },
    ];
    const rows = toolActivities(events, false);
    expect(rows.map((row) => [row.id, row.status, row.content])).toEqual([
      ["a", "interrupted", ""],
      ["b", "completed", "Python 3.13"],
    ]);
    expect(rows[0]!.arguments).toBe("{broken");
  });

  it("keeps document/chunk/version/page provenance when search and read_context return the same source", () => {
    const header =
      "[1] document=doc-1 chunk=chunk-1 product=M9F6820 version=V1.7 pages=113-113 section=12.1 ADC kind=TEXT score=0.8\n";
    const evidence = evidenceItems([
      { content: header + "短摘录" },
      { content: header + "展开后的完整原文依据" },
      {
        content:
          header.replace("doc-1", "doc-2").replace("V1.7", "V2.0") + "另一文档",
      },
    ]);
    expect(evidence).toHaveLength(2);
    expect(evidence[0]).toEqual({
      document: "doc-1",
      chunk: "chunk-1",
      product: "M9F6820",
      version: "V1.7",
      pages: "113",
      section: "12.1 ADC",
      text: "展开后的完整原文依据",
    });
    expect(evidence[1]!.version).toBe("V2.0");
  });

  it("shows only the selected file diff with correct old/new line numbers", () => {
    const patch =
      "--- before/a.c\n+++ after/a.c\n@@ -1 +1 @@\n-int a = 1;\n+int a = 2;\n--- before/b.c\n+++ after/b.c\n@@ -0,0 +1 @@\n+int b = 3;\n";
    const lines = fileDiff(patch, "b.c");
    expect(lines.filter((line) => line.kind !== "hunk")).toEqual([
      { kind: "added", text: "+int b = 3;", old: null, next: 1 },
    ]);
    expect(
      fileDiff(patch, "a.c").some((line) => line.text.includes("int b")),
    ).toBe(false);
  });

  it("keeps streamed reasoning, finalized reasoning and tools in runtime step order without duplication", () => {
    const events: TaskEvent[] = [
      {
        sequence: 1,
        type: "AssistantReasoningDelta",
        payload: { turn: 1, step: 1, text: "先查" },
      },
      {
        sequence: 2,
        type: "AssistantReasoningDelta",
        payload: { turn: 1, step: 1, text: "手册" },
      },
      {
        sequence: 3,
        type: "AssistantTurnEvent",
        payload: {
          turn: 1,
          step: 1,
          message: { reasoning_content: "先查手册" },
        },
      },
      {
        sequence: 4,
        type: "ToolStarted",
        payload: { turn: 1, step: 1, call: { id: "a", name: "search_spec" } },
      },
      { sequence: 5, type: "StepFinished", payload: { turn: 1, step: 1 } },
      {
        sequence: 6,
        type: "AssistantTurnEvent",
        payload: {
          turn: 1,
          step: 2,
          message: { reasoning_content: "工具返回后再核对配置" },
        },
      },
    ];
    const rows = traceSteps(events);
    expect(rows.map((row) => row.reasoning)).toEqual([
      "先查手册",
      "工具返回后再核对配置",
    ]);
    expect(rows[0]!.finished).toBe(true);
    expect(rows[0]!.events[0]!.type).toBe("ToolStarted");
  });
});
