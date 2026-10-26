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

  it("loads a recheck without rewriting the original failed task", async () => {
    const task = useTaskStore();
    task.taskId = "task-1";
    task.state = "failed";
    task.result = { verification: { status: "failed" } };
    const delivery = { task_acceptance: "failed", recheck: { runs: [{ passed: true }] } };
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ passed: true })))
      .mockResolvedValueOnce(new Response(JSON.stringify(delivery)));
    vi.stubGlobal("fetch", fetch);
    await task.recheck();
    expect(fetch.mock.calls[0]![0]).toBe("/api/agent/tasks/task-1/recheck");
    expect(fetch.mock.calls[0]![1].method).toBe("POST");
    expect(task.delivery).toEqual(delivery);
    expect(task.state).toBe("failed");
    expect(task.result.verification!.status).toBe("failed");
    expect(task.recheckingTaskId).toBeNull();
  });

  it("does not apply a late recheck to a different selected task", async () => {
    const task = useTaskStore();
    task.taskId = "old";
    let finish!: (response: Response) => void;
    vi.stubGlobal("fetch", vi.fn()
      .mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ changed_files: ["old.c"] }))));
    const pending = task.recheck();
    task.reset();
    task.taskId = "new";
    finish(new Response(JSON.stringify({ passed: true })));
    await pending;
    expect(task.delivery).toBeNull();
    expect(task.recheckError).toBeNull();
  });

  it("reloads undo history while preserving the original result", async () => {
    const task = useTaskStore();
    task.taskId = "task-1";
    task.state = "completed";
    task.result = { answer: "original answer" };
    const delivery = { undo: { available: false, runs: [{ status: "completed" }] } };
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "completed" })))
      .mockResolvedValueOnce(new Response(JSON.stringify(delivery)));
    vi.stubGlobal("fetch", fetch);
    await task.undo();
    expect(fetch.mock.calls[0]![0]).toBe("/api/agent/tasks/task-1/undo");
    expect(fetch.mock.calls[0]![1].method).toBe("POST");
    expect(task.delivery).toEqual(delivery);
    expect(task.result.answer).toBe("original answer");
    expect(task.state).toBe("completed");
    expect(task.undoingTaskId).toBeNull();
  });

  it("shows a conflict and refreshes undo availability", async () => {
    const task = useTaskStore();
    task.taskId = "task-1";
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "module.c 已改变" }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ undo: { available: false } })));
    vi.stubGlobal("fetch", fetch);
    await task.undo();
    expect(task.undoError).toBe("module.c 已改变");
    expect(task.delivery!.undo!.available).toBe(false);
  });

  it("does not overlap undo with rechecks or update a different selected task", async () => {
    const task = useTaskStore();
    task.taskId = "old";
    let finish!: (response: Response) => void;
    const fetch = vi.fn()
      .mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve; }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ changed_files: ["old.c"] })));
    vi.stubGlobal("fetch", fetch);
    const pending = task.undo();
    await task.recheck();
    await task.undo();
    expect(fetch).toHaveBeenCalledOnce();
    task.reset();
    task.taskId = "new";
    finish(new Response(JSON.stringify({ status: "completed" })));
    await pending;
    expect(task.delivery).toBeNull();
    expect(task.undoError).toBeNull();
  });
});
