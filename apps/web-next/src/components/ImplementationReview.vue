<script setup lang="ts">
import { computed } from "vue";
import type { DecisionEvidence, ImplementationDecision } from "../stores/task";
import { readableOutput } from "../utils/presentation";
import AppIcon from "./AppIcon.vue";
import MarkdownContent from "./MarkdownContent.vue";

const props = defineProps<{
  decisions: ImplementationDecision[];
  changedFiles: string[];
}>();
defineEmits<{ file: [path: string]; source: [ref: DecisionEvidence] }>();
const unlinked = computed(() =>
  props.changedFiles.filter(
    (path) =>
      !props.decisions.some((item) =>
        item.code_refs.some((ref) => ref.path === path),
      ),
  ),
);
const codeStatus = {
  located: "位置已核对",
  changed: "记录后源码有变化",
  missing: "文件已不存在",
};
const checkStatus = {
  passed: "执行通过 · 版本一致",
  failed: "调用失败",
  outdated: "执行后源码有变化",
  unverified: "未取得通过结果",
};
</script>

<template>
  <div class="implementation-review">
    <p class="panel-description">
      说明由模型整理，定位与检查状态来自实际记录。引用可定位不代表技术结论已经验证。
    </p>
    <div v-if="!decisions.length" class="panel-empty implementation-empty">
      <AppIcon name="book" :size="26" />
      <h3>本轮尚未记录实现依据</h3>
      <p>
        新任务可关联需求、源码、规格与检查。已有任务的完整差异和执行记录仍可查看。
      </p>
    </div>
    <article
      v-for="item in decisions"
      :key="item.id"
      class="implementation-decision"
      data-testid="implementation-decision"
    >
      <h3>{{ item.title }}</h3>
      <div class="decision-requirement">
        <span>需求</span>{{ item.requirement }}
      </div>
      <MarkdownContent :content="item.rationale" />
      <div class="decision-section-label">
        <AppIcon name="code" :size="14" />源码位置
      </div>
      <div
        v-for="(ref, index) in item.code_refs"
        :key="index"
        class="decision-code"
      >
        <div class="decision-code-heading">
          <code>{{ ref.path }}:{{ ref.start_line }}–{{ ref.end_line }}</code>
          <span :class="['reference-state', ref.status]">{{
            codeStatus[ref.status]
          }}</span>
        </div>
        <pre>{{ ref.excerpt }}</pre>
        <button
          v-if="changedFiles.includes(ref.path)"
          class="decision-text-button"
          @click="$emit('file', ref.path)"
        >
          查看文件差异<AppIcon name="arrow" :size="12" />
        </button>
      </div>
      <div class="decision-section-label">
        <AppIcon name="book" :size="14" />规格依据
      </div>
      <p v-if="!item.evidence_refs.length" class="decision-no-evidence">
        未关联本轮规格引用。
      </p>
      <button
        v-for="ref in item.evidence_refs"
        :key="`${ref.document_id}:${ref.chunk_id}`"
        class="decision-source"
        @click="$emit('source', ref)"
      >
        <span
          >{{ ref.product }} <span>{{ ref.version }}</span> · p.{{
            ref.pages
          }}</span
        >
        <strong>{{ ref.section || "规格书片段" }}</strong
        ><AppIcon name="chevron" :size="13" />
      </button>
      <div class="decision-section-label">
        <AppIcon name="check" :size="14" />关联检查
      </div>
      <p v-if="!item.checks.length" class="decision-no-evidence">
        未关联检查，不推断为已验证。
      </p>
      <details
        v-for="check in item.checks"
        :key="check.call_id"
        class="decision-check"
      >
        <summary>
          <span>{{
            check.name === "verify_project" ? "宿主检查" : "运行命令"
          }}</span
          ><span :class="['reference-state', check.status]">{{
            checkStatus[check.status]
          }}</span
          ><AppIcon name="chevron" :size="12" />
        </summary>
        <pre class="tool-output">{{ readableOutput(check.content) }}</pre>
        <code class="decision-call-id">{{ check.call_id }}</code>
      </details>
      <div v-if="item.limitations?.length" class="decision-limitations">
        <strong>待确认与验证范围</strong>
        <ul>
          <li v-for="(limit, index) in item.limitations" :key="index">
            {{ limit }}
          </li>
        </ul>
      </div>
    </article>
    <div v-if="unlinked.length" class="unlinked-changes">
      <div class="panel-section-title">
        未关联说明的改动 <span>{{ unlinked.length }} 个文件</span>
      </div>
      <button v-for="file in unlinked" :key="file" @click="$emit('file', file)">
        <AppIcon name="file" :size="14" /><code>{{ file }}</code
        ><AppIcon name="arrow" :size="12" />
      </button>
    </div>
  </div>
</template>
