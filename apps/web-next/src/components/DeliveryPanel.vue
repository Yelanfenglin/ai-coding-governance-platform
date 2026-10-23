<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import type { DecisionEvidence, Delivery, TaskResponse } from "../stores/task";
import {
  evidenceItems,
  fileDiff,
  objectValue,
  readableOutput,
  verificationLabel,
} from "../utils/presentation";
import AppIcon from "./AppIcon.vue";
import MarkdownContent from "./MarkdownContent.vue";
import ImplementationReview from "./ImplementationReview.vue";

const props = defineProps<{
  delivery: Delivery | null;
  result: TaskResponse | null;
}>();
defineEmits<{ close: [] }>();
const tab = defineModel<string>("tab", { default: "changes" });
const selectedFile = ref("");
const selectedSource = ref("");
function openFile(path: string) {
  selectedFile.value = path;
  tab.value = "changes";
}
async function openSource(ref: DecisionEvidence) {
  selectedSource.value = `${ref.document_id}:${ref.chunk_id}`;
  tab.value = "sources";
  await nextTick();
  document
    .getElementById(`evidence-${selectedSource.value}`)
    ?.scrollIntoView({ block: "nearest" });
}
watch(
  () => props.delivery,
  (value) => {
    if (!value?.changed_files.includes(selectedFile.value))
      selectedFile.value = value?.changed_files[0] ?? "";
  },
  { immediate: true },
);
const evidence = computed(() => evidenceItems(props.delivery?.evidence ?? []));
const verification = computed(
  () => props.delivery?.verification ?? props.result?.verification,
);
const runs = computed(
  () =>
    (objectValue(verification.value).runs ?? []) as Array<
      Record<string, unknown>
    >,
);
const diff = computed(() => {
  try {
    return fileDiff(props.delivery?.diff ?? "", selectedFile.value);
  } catch {
    return [];
  } // Preserve the exact delivered patch below when it is not a parseable diff.
});
const tabs = computed(() => [
  {
    id: "implementation",
    label: "实现",
    count: props.delivery?.decisions?.length ?? 0,
  },
  {
    id: "changes",
    label: "改动",
    count: props.delivery?.changed_files.length ?? 0,
  },
  { id: "sources", label: "依据", count: evidence.value.length },
  { id: "checks", label: "验收", count: null },
]);
const command = (row: Record<string, unknown>) => {
  const args = objectValue(row.arguments);
  return Array.isArray(args.command)
    ? args.command.join(" ")
    : JSON.stringify(row.arguments);
};
</script>

<template>
  <aside class="inspector" aria-label="工程审阅">
    <header class="inspector-heading">
      <div><AppIcon name="panel" :size="17" /><strong>工程审阅</strong></div>
      <button
        class="icon-button"
        aria-label="收起审阅面板"
        @click="$emit('close')"
      >
        <AppIcon name="close" :size="17" />
      </button>
    </header>
    <div class="inspector-tabs" role="tablist" aria-label="交付内容">
      <button
        v-for="item in tabs"
        :id="`tab-${item.id}`"
        :key="item.id"
        role="tab"
        :aria-selected="tab === item.id"
        :aria-controls="`panel-${item.id}`"
        :class="{ selected: tab === item.id }"
        @click="tab = item.id"
      >
        {{ item.label }}<span v-if="item.count !== null">{{ item.count }}</span>
      </button>
    </div>
    <section
      :id="`panel-${tab}`"
      class="inspector-content"
      role="tabpanel"
      :aria-labelledby="`tab-${tab}`"
    >
      <ImplementationReview
        v-if="tab === 'implementation'"
        :decisions="delivery?.decisions ?? []"
        :changed-files="delivery?.changed_files ?? []"
        @file="openFile"
        @source="openSource"
      />
      <template v-else-if="tab === 'changes'">
        <div v-if="!delivery" class="panel-empty">
          <AppIcon name="file" :size="28" />
          <h3>改动会显示在这里</h3>
          <p>任务结束后，按文件查看这次实际写入的内容。</p>
        </div>
        <div v-else-if="!delivery.changed_files.length" class="panel-empty">
          <AppIcon name="check" :size="28" />
          <h3>本轮没有文件改动</h3>
          <p>这次任务的执行记录仍保留在对话中。</p>
        </div>
        <template v-else>
          <div class="panel-section-title">
            本轮修改 <span>{{ delivery.changed_files.length }} 个文件</span>
          </div>
          <div class="changed-files">
            <button
              v-for="file in delivery.changed_files"
              :key="file"
              :title="file"
              :class="{ selected: file === selectedFile }"
              @click="selectedFile = file"
            >
              <AppIcon name="file" :size="16" /><span>{{ file }}</span
              ><span class="file-modified">改</span>
            </button>
          </div>
          <div class="diff-heading">
            <code>{{ selectedFile }}</code
            ><span>统一差异</span>
          </div>
          <div
            v-if="diff.length"
            class="diff-view"
            data-testid="diff"
            tabindex="0"
            aria-label="文件差异"
          >
            <div
              v-for="(line, index) in diff"
              :key="index"
              :class="['diff-line', line.kind]"
            >
              <span class="line-number">{{ line.old }}</span
              ><span class="line-number">{{ line.next }}</span
              ><code>{{ line.text }}</code>
            </div>
          </div>
          <pre v-else class="raw-patch" data-testid="diff">{{
            delivery.diff
          }}</pre>
        </template>
      </template>
      <template v-else-if="tab === 'sources'">
        <div v-if="!evidence.length" class="panel-empty">
          <AppIcon name="book" :size="28" />
          <h3>还没有规格引用</h3>
          <p>检索与展开过的证据会保留文档、版本、页码和片段定位。</p>
        </div>
        <template v-else>
          <p class="panel-description">
            本轮工具实际返回的证据，按文档与片段去重。
          </p>
          <details
            v-for="item in evidence"
            :key="`${item.document}:${item.chunk}`"
            :id="`evidence-${item.document}:${item.chunk}`"
            :open="selectedSource === `${item.document}:${item.chunk}`"
            class="evidence-item"
          >
            <summary>
              <span class="evidence-meta"
                ><span>{{ item.product }}</span
                ><span>{{ item.version }}</span
                ><span>p.{{ item.pages }}</span></span
              >
              <strong>{{ item.section || "规格书片段" }}</strong
              ><AppIcon name="chevron" :size="14" class="disclosure-arrow" />
            </summary>
            <div class="evidence-text">
              <MarkdownContent :content="item.text" />
              <dl class="evidence-location">
                <dt>文档</dt>
                <dd>{{ item.document }}</dd>
                <dt>片段</dt>
                <dd>{{ item.chunk }}</dd>
              </dl>
            </div>
          </details>
        </template>
        <details v-if="delivery?.evidence.length" class="raw-details">
          <summary>原始检索记录</summary>
          <pre>{{ JSON.stringify(delivery.evidence, null, 2) }}</pre>
        </details>
      </template>
      <template v-else>
        <div
          :class="['verification-card', verification?.status]"
          data-testid="verification"
        >
          <AppIcon
            :name="
              verification?.status === 'passed'
                ? 'check'
                : verification?.status === 'failed'
                  ? 'warning'
                  : 'clock'
            "
            :size="22"
          />
          <div>
            <h3>{{ verificationLabel(verification?.status) }}</h3>
            <p>模型的完成答复与宿主检查分别记录。</p>
          </div>
        </div>
        <details v-for="(run, index) in runs" :key="index" class="check-run">
          <summary>
            <AppIcon :name="run.passed ? 'check' : 'warning'" :size="15" />{{
              run.phase === "delivery" ? "最终交付检查" : "开发过程检查"
            }}<span>退出码 {{ run.exit_code }}</span>
          </summary>
          <pre class="command-display">{{
            Array.isArray(run.command) ? run.command.join(" ") : run.command
          }}</pre>
          <pre class="tool-output">{{ run.output }}</pre>
        </details>
        <div v-if="delivery?.commands.length" class="command-records">
          <div class="panel-section-title">实际运行的命令</div>
          <details
            v-for="(row, index) in delivery.commands"
            :key="index"
            class="check-run"
          >
            <summary>
              <AppIcon name="terminal" :size="15" /><span>{{
                row.is_error ? "执行报错" : "命令已返回"
              }}</span
              ><span v-if="objectValue(row.content).exit_code !== undefined"
                >退出码 {{ objectValue(row.content).exit_code }}</span
              >
            </summary>
            <pre class="command-display">$ {{ command(row) }}</pre>
            <pre class="tool-output">{{
              readableOutput(String(row.content ?? ""))
            }}</pre>
            <p class="panel-description">
              {{
                row.matches_delivery
                  ? "命令执行时的文件版本与最终交付一致。"
                  : "这条命令执行后文件版本有变化，不能据此认定最终版本通过。"
              }}
            </p>
          </details>
        </div>
      </template>
    </section>
    <footer class="inspector-footer">
      <AppIcon name="folder" :size="14" />当前工程 · 本轮交付
    </footer>
  </aside>
</template>
