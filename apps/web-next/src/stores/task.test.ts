import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useTaskStore } from "./task";

describe("local Agent task store", () => {
  beforeEach(() => { setActivePinia(createPinia()); vi.unstubAllGlobals(); });

  it("deduplicates replay and waits for host completion after the model turn", () => {
    const task = useTaskStore();
    task.apply({ sequence: 1, type: "TaskStarted", payload: {} });
    const delta = { sequence: 2, type: "AssistantTextDelta", payload: { text: "working" } };
    task.apply(delta);
    task.apply(delta);
    task.apply({ sequence: 3, type: "TurnFinished", payload: { status: "completed" } });
    expect(task.answer).toBe("working");
    expect(task.state).toBe("running");
    task.apply({ sequence: 4, type: "TaskFinished", payload: {
      status: "failed", exit_code: 5,
      response: { status: "completed", answer: "done", verification: { status: "failed" } },
    } });
    expect(task.state).toBe("failed");
    expect(task.answer).toBe("done");
    expect(task.events).toHaveLength(4);
  });

  it("keeps cancellation pending until the runtime has actually stopped", async () => {
    const task = useTaskStore();
    task.taskId = "task-1";
    task.state = "running";
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ status: "cancelling" }), { status: 202 },
    )));
    await task.cancel();
    expect(task.state).toBe("cancelling");
    expect(task.active).toBe(true);
    task.apply({ sequence: 1, type: "TaskFinished", payload: {
      status: "cancelled", response: { status: "aborted" },
    } });
    expect(task.active).toBe(false);
  });

  it("connects only to the local runtime contract and closes the final event stream", async () => {
    const close = vi.fn();
    const streams: Array<{ onmessage: ((event: { data: string }) => void) | null }> = [];
    const urls: string[] = [];
    vi.stubGlobal("EventSource", class {
      onmessage = null;
      onerror = null;
      onopen = null;
      close = close;
      constructor(url: string) { urls.push(url); streams.push(this); }
    });
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      id: "task-1", status: "pending",
    }), { status: 201 }));
    vi.stubGlobal("fetch", fetch);
    const task = useTaskStore();
    await task.create("conversation-1", "write a module");
    expect(urls).toEqual(["/api/agent/tasks/task-1/events?after=0"]);
    const body = JSON.parse(fetch.mock.calls[0]![1].body);
    expect(Object.keys(body).sort()).toEqual(["conversation_id", "idempotency_key", "prompt", "skill"]);
    expect(body.skill).toBeNull();
    streams[0]!.onmessage!({ data: JSON.stringify({ sequence: 1, type: "TaskFinished",
      payload: { status: "completed", response: { answer: "done" } } }) });
    expect(task.answer).toBe("done");
    expect(close).toHaveBeenCalledOnce();
  });

  it("submits the selected skill and restores its label from task history", async () => {
    const task = useTaskStore();
    vi.spyOn(task, "connect").mockImplementation(() => {});
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      id: "task-2", status: "pending", skill: "module-development", skill_title: "模块开发",
    }), { status: 201 }));
    vi.stubGlobal("fetch", fetch);
    await task.create("conversation-1", "实现模块", "module-development");
    expect(JSON.parse(fetch.mock.calls[0]![1].body).skill).toBe("module-development");
    expect(task.skillTitle).toBe("模块开发");
    task.reset();
    expect(task.skillTitle).toBeNull();
  });
});
