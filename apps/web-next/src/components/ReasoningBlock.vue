<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
import AppIcon from "./AppIcon.vue";
import MarkdownContent from "./MarkdownContent.vue";

const props = defineProps<{
  content: string;
  thinking: boolean;
  step: number;
}>();
const open = ref(props.thinking);
const body = ref<HTMLElement>();
let follow = true;
function trackScroll() {
  if (body.value)
    follow =
      body.value.scrollHeight - body.value.scrollTop - body.value.clientHeight <
      60;
}
watch(
  () => props.thinking,
  (value) => {
    open.value = value;
  },
);
watch(
  () => props.content,
  async () => {
    await nextTick();
    if (props.thinking && follow && body.value)
      body.value.scrollTop = body.value.scrollHeight;
  },
);
</script>

<template>
  <details
    class="reasoning-block"
    :open="open"
    @toggle="open = ($event.target as HTMLDetailsElement).open"
  >
    <summary>
      <AppIcon name="chip" :size="15" /><span>{{
        thinking ? "思考中" : "思考过程"
      }}</span
      ><span class="reasoning-step">第 {{ step }} 步</span
      ><span v-if="thinking" class="activity-dot"></span
      ><AppIcon name="chevron" :size="12" class="disclosure-arrow" />
    </summary>
    <div
      ref="body"
      class="reasoning-content"
      data-testid="reasoning"
      @scroll="trackScroll"
    >
      <MarkdownContent :content="content" />
    </div>
  </details>
</template>
