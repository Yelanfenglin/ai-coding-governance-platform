<script setup lang="ts">
import { computed } from "vue";
import type { TaskEvent } from "../stores/task";
import { traceSteps } from "../utils/presentation";
import ReasoningBlock from "./ReasoningBlock.vue";
import ToolActivity from "./ToolActivity.vue";

const props = defineProps<{ events: TaskEvent[]; active: boolean }>();
const steps = computed(() => traceSteps(props.events));
</script>

<template>
  <div v-if="steps.length" class="execution-trace">
    <div v-for="step in steps" :key="step.id" class="trace-step">
      <ReasoningBlock
        v-if="step.reasoning"
        :content="step.reasoning"
        :step="step.step"
        :thinking="active && !step.assistantFinished && !step.finished"
      />
      <ToolActivity :events="step.events" :active="active && !step.finished" />
    </div>
  </div>
</template>
