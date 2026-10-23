import { defineStore } from "pinia";
import { apiBase, apiRequest } from "../api";

export type TaskState =
  | "idle"
  | "pending"
  | "running"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled"
  | "budget_exceeded";

export interface TaskResponse {
  status?: string;
  answer?: string;
  error?: string | null;
  steps?: number;
  usage?: { input_tokens: number; output_tokens: number; total_tokens: number };
  verification?: { status: string; reason?: string };
  delivery?: string;
}

export interface TaskRecord {
  id: string;
  conversation_id: string;
  prompt: string;
  status: TaskState;
  created_at: string;
  response: TaskResponse | null;
  exit_code: number | null;
}

export interface TaskEvent {
  sequence: number;
  type: string;
  payload: Record<string, unknown>;
}

export interface Delivery {
  changed_files: string[];
  diff: string;
  commands: Array<Record<string, unknown>>;
  evidence: Array<Record<string, unknown>>;
  verification: { status: string };
  task_acceptance: string;
  decisions?: ImplementationDecision[] | null;
}

export interface DecisionEvidence {
  document_id: string;
  chunk_id: string;
  product: string;
  version: string;
  pages: string;
  section: string;
  call_id: string;
}

export interface ImplementationDecision {
  id: string;
  title: string;
  requirement: string;
  rationale: string;
  code_refs: Array<{
    path: string;
    start_line: number;
    end_line: number;
    excerpt: string;
    status: "located" | "changed" | "missing";
  }>;
  evidence_refs: DecisionEvidence[];
  checks: Array<{
    call_id: string;
    name: string;
    content: string;
    matches_delivery: boolean;
    status: "passed" | "failed" | "outdated" | "unverified";
  }>;
  limitations?: string[];
}

export function stateAfterEvent(
  current: TaskState,
  event: TaskEvent,
): TaskState {
  if (event.type === "TaskStarted") return "running";
  // TurnFinished precedes host verification and delivery. Only TaskFinished closes a task.
  if (event.type === "TaskFinished") return event.payload.status as TaskState;
  return current;
}

export const useTaskStore = defineStore("task", {
  state: () => ({
    taskId: null as string | null,
    state: "idle" as TaskState,
    events: [] as TaskEvent[],
    lastSequence: 0,
    stream: null as EventSource | null,
    error: null as string | null,
    answer: "",
    result: null as TaskResponse | null,
    delivery: null as Delivery | null,
  }),
  getters: {
    active: (state) =>
      ["pending", "running", "cancelling"].includes(state.state),
  },
  actions: {
    apply(event: TaskEvent) {
      if (event.sequence <= this.lastSequence) return;
      this.events.push(event);
      this.lastSequence = event.sequence;
      this.state = stateAfterEvent(this.state, event);
      if (event.type === "AssistantTextDelta")
        this.answer += String(event.payload.text ?? "");
      if (event.type === "TaskFinished") {
        this.result = (event.payload.response ?? {}) as TaskResponse;
        this.answer = this.result.answer ?? this.answer;
        this.error = this.result.error ?? null;
        this.disconnect();
        if (this.result.delivery && this.taskId)
          void this.loadDelivery(this.taskId);
      }
    },
    reset() {
      this.disconnect();
      this.taskId = null;
      this.state = "idle";
      this.events = [];
      this.lastSequence = 0;
      this.error = null;
      this.answer = "";
      this.result = null;
      this.delivery = null;
    },
    async create(conversationId: string, prompt: string) {
      const record = await apiRequest<TaskRecord>("/tasks", {
        method: "POST",
        body: JSON.stringify({
          conversation_id: conversationId,
          prompt,
          idempotency_key: crypto.randomUUID(),
        }),
      });
      this.attach(record);
      return record;
    },
    attach(record: TaskRecord) {
      this.reset();
      this.taskId = record.id;
      this.state = record.status;
      this.connect();
    },
    connect() {
      if (!this.taskId) return;
      this.disconnect();
      const source = new EventSource(
        `${apiBase}/tasks/${this.taskId}/events?after=${this.lastSequence}`,
        { withCredentials: true },
      );
      source.onmessage = (message) => {
        try {
          this.apply(JSON.parse(message.data) as TaskEvent);
        } catch {
          this.error = "收到无法解析的任务事件";
        }
      };
      source.onopen = () => {
        this.error = null;
      };
      source.onerror = () => {
        this.error = "事件流暂时中断，浏览器将自动续接";
      };
      this.stream = source;
    },
    async cancel() {
      if (!this.taskId) return;
      const record = await apiRequest<TaskRecord>(
        `/tasks/${this.taskId}/cancel`,
        { method: "POST" },
      );
      this.state = record.status;
    },
    async loadDelivery(identifier: string) {
      try {
        const delivery = await apiRequest<Delivery>(
          `/tasks/${identifier}/delivery`,
        );
        if (this.taskId === identifier) this.delivery = delivery;
      } catch (reason) {
        if (this.taskId === identifier) {
          this.error =
            reason instanceof Error ? reason.message : "交付记录读取失败";
        }
      }
    },
    disconnect() {
      this.stream?.close();
      this.stream = null;
    },
  },
});
