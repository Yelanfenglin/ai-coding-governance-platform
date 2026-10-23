<script setup lang="ts">
import { onBeforeUnmount, ref } from "vue";
import { apiBase } from "../api";
import type { TaskEvent } from "../stores/task";
import ExecutionTrace from "./ExecutionTrace.vue";
const props = defineProps<{ taskId: string }>();
const events = ref<TaskEvent[]>([]);
const loaded = ref(false);
const loading = ref(false);
const error = ref("");
let stream: EventSource | null = null;
function expand(event: Event) {
  if (
    !(event.target as HTMLDetailsElement).open ||
    loaded.value ||
    loading.value
  )
    return;
  loading.value = true;
  stream = new EventSource(`${apiBase}/tasks/${props.taskId}/events`, {
    withCredentials: true,
  });
  stream.onmessage = (message) => {
    const item = JSON.parse(message.data) as TaskEvent;
    if (item.sequence > (events.value.at(-1)?.sequence ?? 0))
      events.value.push(item);
    if (item.type === "TaskFinished") {
      loaded.value = true;
      loading.value = false;
      error.value = "";
      stream?.close();
    }
  };
  stream.onerror = () => {
    error.value = "过程记录暂时无法读取，正在重连…";
  };
}
onBeforeUnmount(() => stream?.close());
</script>

<template>
  <details class="history-trace" @toggle="expand">
    <summary>查看这轮思考与执行过程</summary>
    <p v-if="loading" class="muted">正在读取记录…</p>
    <p v-if="error" class="muted">{{ error }}</p>
    <ExecutionTrace :events="events" :active="false" />
    <p
      v-if="
        loaded &&
        !events.some(
          (item) =>
            item.type === 'AssistantReasoningDelta' ||
            item.type === 'ToolStarted',
        )
      "
      class="muted"
    >
      本轮没有返回思考或工具记录。
    </p>
  </details>
</template>
