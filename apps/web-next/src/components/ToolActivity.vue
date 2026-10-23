<script setup lang="ts">
import { computed } from "vue";
import type { TaskEvent } from "../stores/task";
import {
  toolActivities,
  toolLabel,
  toolIcon,
  toolSubject,
  readableOutput,
  objectValue,
} from "../utils/presentation";
import AppIcon from "./AppIcon.vue";
const props = defineProps<{ events: TaskEvent[]; active: boolean }>();
const tools = computed(() => toolActivities(props.events, props.active));
const current = computed(() =>
  tools.value.find((tool) => tool.status === "running"),
);
const lead = computed(
  () =>
    current.value ?? (tools.value.length === 1 ? tools.value[0] : undefined),
);
const failures = computed(
  () => tools.value.filter((tool) => tool.status === "failed").length,
);
const interrupted = computed(
  () => tools.value.filter((tool) => tool.status === "interrupted").length,
);
const statuses = {
  running: "进行中",
  completed: "完成",
  failed: "失败",
  interrupted: "已中断",
};
</script>

<template>
  <details v-if="tools.length" class="tool-group">
    <summary class="tool-group-summary">
      <span class="tool-group-icon"><AppIcon name="code" :size="16" /></span>
      <span>{{
        lead ? toolLabel(lead.name) : `执行了 ${tools.length} 项操作`
      }}</span>
      <span v-if="lead" class="tool-current-file">{{ toolSubject(lead) }}</span>
      <span v-if="failures" class="tool-error-count"
        >{{ failures }} 项报错</span
      >
      <span v-else-if="interrupted" class="tool-error-count"
        >{{ interrupted }} 项中断</span
      >
      <span v-else-if="!current" class="tool-group-check"
        ><AppIcon name="check" :size="14"
      /></span>
      <AppIcon name="chevron" :size="14" class="disclosure-arrow" />
    </summary>
    <div class="tool-list">
      <details v-for="tool in tools" :key="tool.id" class="tool-row">
        <summary>
          <AppIcon :name="toolIcon(tool.name)" :size="16" />
          <span class="tool-name">{{ toolLabel(tool.name) }}</span>
          <span class="tool-subject">{{ toolSubject(tool) }}</span>
          <span :class="['tool-result-state', tool.status]">{{
            statuses[tool.status]
          }}</span>
          <AppIcon name="chevron" :size="12" class="disclosure-arrow" />
        </summary>
        <div class="tool-detail">
          <div v-if="tool.name === 'run_command'" class="command-display">
            <span>$</span> {{ toolSubject(tool) }}
          </div>
          <pre v-if="tool.content" class="tool-output">{{
            readableOutput(tool.content)
          }}</pre>
          <p
            v-if="objectValue(tool.content).exit_code !== undefined"
            class="tool-exit"
          >
            退出码 {{ objectValue(tool.content).exit_code }}
          </p>
          <details class="raw-details">
            <summary>参数与调用标识</summary>
            <code>{{ tool.id }}</code>
            <pre>{{
              typeof tool.arguments === "string"
                ? tool.arguments
                : JSON.stringify(tool.arguments, null, 2)
            }}</pre>
          </details>
        </div>
      </details>
    </div>
  </details>
</template>
