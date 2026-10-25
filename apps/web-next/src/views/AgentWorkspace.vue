<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { useTaskStore, type TaskRecord } from "../stores/task";
import { useResourceStore } from "../stores/resources";
import {
  toolActivities,
  toolLabel,
  verificationLabel,
} from "../utils/presentation";
import AppIcon from "../components/AppIcon.vue";
import MarkdownContent from "../components/MarkdownContent.vue";
import ExecutionTrace from "../components/ExecutionTrace.vue";
import HistoryTrace from "../components/HistoryTrace.vue";
import DeliveryPanel from "../components/DeliveryPanel.vue";

const task = useTaskStore();
const resources = useResourceStore();
const projectId = ref("");
const conversationId = ref("");
const prompt = ref("");
const selectedSkill = ref("");
const currentPrompt = ref("");
const submitting = ref(false);
const loadingHistory = ref(false);
const formError = ref("");
const history = ref<TaskRecord[]>([]);
const sidebarOpen = ref(false);
const inspectorOpen = ref(false);
const reviewTab = ref("changes");
const search = ref("");
const copied = ref(false);
const transcript = ref<HTMLElement>();
const input = ref<HTMLTextAreaElement>();
const followBottom = ref(true);
const project = computed(() =>
  resources.projects.find((item) => item.id === projectId.value),
);
const conversation = computed(() =>
  resources.conversations.find((item) => item.id === conversationId.value),
);
const conversations = computed(() =>
  resources.conversations.filter(
    (item) =>
      item.project_id === projectId.value &&
      item.title.toLowerCase().includes(search.value.toLowerCase()),
  ),
);
const earlier = computed(() =>
  history.value.filter((item) => item.id !== task.taskId),
);
const hasConversation = computed(
  () => earlier.value.length > 0 || !!currentPrompt.value,
);
const canSend = computed(
  () =>
    !!projectId.value &&
    !!prompt.value.trim() &&
    !task.active &&
    !task.recheckingTaskId &&
    !submitting.value &&
    !loadingHistory.value,
);
const tools = computed(() => toolActivities(task.events, task.active));
const currentTool = computed(() =>
  tools.value.find((item) => item.status === "running"),
);
const statusLabels = {
  idle: "就绪",
  pending: "等待执行",
  running: "执行中",
  cancelling: "正在取消",
  completed: "执行完成",
  failed: "执行失败",
  cancelled: "已取消",
  budget_exceeded: "预算耗尽",
};
const activityLabel = computed(() =>
  task.state === "cancelling"
    ? "正在取消，请稍候"
    : currentTool.value
      ? toolLabel(currentTool.value.name)
      : task.answer
        ? "正在整理答复与交付"
        : "正在分析需求",
);
function showError(reason: unknown) {
  formError.value = reason instanceof Error ? reason.message : "请求失败";
}
function trackScroll() {
  const area = transcript.value;
  if (area)
    followBottom.value =
      area.scrollHeight - area.scrollTop - area.clientHeight < 100;
}
async function scrollToBottom() {
  await nextTick();
  if (followBottom.value && transcript.value)
    transcript.value.scrollTop = transcript.value.scrollHeight;
}
function resizeInput() {
  if (!input.value) return;
  input.value.style.height = "auto";
  input.value.style.height = `${Math.min(input.value.scrollHeight, 180)}px`;
}
function newConversation() {
  if (task.active || submitting.value) return;
  conversationId.value = "";
  loadingHistory.value = false;
  task.reset();
  history.value = [];
  currentPrompt.value = "";
  prompt.value = "";
  selectedSkill.value = "";
  formError.value = "";
  inspectorOpen.value = false;
  sidebarOpen.value = false;
  localStorage.removeItem("agent-conversation");
  void nextTick(() => {
    resizeInput();
    input.value?.focus();
  });
}
async function selectConversation(identifier: string) {
  if (task.active || submitting.value) return;
  conversationId.value = identifier;
  sidebarOpen.value = false;
  task.reset();
  history.value = [];
  currentPrompt.value = "";
  prompt.value = "";
  selectedSkill.value = "";
  formError.value = "";
  followBottom.value = true;
  loadingHistory.value = true;
  try {
    const detail = await resources.history(identifier);
    if (conversationId.value !== identifier) return;
    history.value = detail.tasks;
    const latest = detail.tasks.at(-1);
    if (latest) {
      currentPrompt.value = latest.prompt;
      task.attach(latest);
    }
    localStorage.setItem("agent-conversation", identifier);
    await scrollToBottom();
  } catch (reason) {
    if (conversationId.value === identifier) showError(reason);
  } finally {
    if (conversationId.value === identifier) loadingHistory.value = false;
  }
}
async function start() {
  if (!canSend.value) return;
  submitting.value = true;
  formError.value = "";
  const question = prompt.value.trim();
  try {
    if (!conversationId.value) {
      const record = await resources.createConversation(
        question.slice(0, 28),
        projectId.value,
      );
      conversationId.value = record.id;
      localStorage.setItem("agent-conversation", record.id);
    }
    await task.create(conversationId.value, question, selectedSkill.value || null);
    currentPrompt.value = question;
    prompt.value = "";
    selectedSkill.value = "";
    copied.value = false;
    followBottom.value = true;
    await nextTick();
    resizeInput();
    await scrollToBottom();
  } catch (reason) {
    showError(reason);
  } finally {
    submitting.value = false;
  }
}
function onKeydown(event: KeyboardEvent) {
  if (
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.isComposing &&
    event.keyCode !== 229
  ) {
    event.preventDefault();
    void start();
  }
}
async function cancel() {
  try {
    await task.cancel();
  } catch (reason) {
    showError(reason);
  }
}
function chooseSkill(name: string) {
  selectedSkill.value = name;
  void nextTick(() => {
    resizeInput();
    input.value?.focus();
  });
}
async function copyAnswer() {
  try {
    await navigator.clipboard.writeText(task.answer);
    copied.value = true;
  } catch (reason) {
    showError(reason);
  }
}
function shortDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
  }).format(new Date(value));
}
watch(
  [() => task.answer, () => task.events.length, () => history.value.length],
  scrollToBottom,
);
watch(
  () => task.state,
  async (state) => {
    if (
      ["completed", "failed", "cancelled", "budget_exceeded"].includes(state) &&
      conversationId.value
    ) {
      const identifier = conversationId.value;
      try {
        const detail = await resources.history(identifier);
        if (identifier === conversationId.value) history.value = detail.tasks;
      } catch (reason) {
        showError(reason);
      }
    }
  },
);
onMounted(async () => {
  try {
    await resources.load();
    projectId.value = resources.projects[0]?.id ?? "";
    const saved = localStorage.getItem("agent-conversation");
    const selected =
      resources.conversations.find((item) => item.id === saved)?.id ??
      resources.conversations[0]?.id;
    if (selected) await selectConversation(selected);
  } catch (reason) {
    showError(reason);
  }
});
onBeforeUnmount(() => task.disconnect());
</script>

<template>
  <div
    :class="[
      'workbench',
      { 'sidebar-open': sidebarOpen, 'inspector-open': inspectorOpen },
    ]"
    @keydown.esc="
      sidebarOpen = false;
      inspectorOpen = false;
    "
  >
    <button
      v-if="sidebarOpen"
      class="sidebar-scrim"
      aria-label="关闭会话导航"
      @click="sidebarOpen = false"
    ></button>
    <aside class="sidebar" aria-label="工程与会话">
      <a class="brand" href="/" @click.prevent="newConversation"
        ><span class="brand-mark"><AppIcon name="chip" :size="23" /></span
        ><span
          ><strong>嵌入式编程助手</strong><small>本地工程工作台</small></span
        ></a
      >
      <button
        class="new-conversation"
        :disabled="task.active || submitting"
        @click="newConversation"
      >
        <AppIcon name="plus" :size="17" />新建会话<span>＋</span>
      </button>
      <div class="sidebar-section-label">当前工程</div>
      <details class="project-switcher">
        <summary>
          <span class="project-symbol"
            ><AppIcon name="folder" :size="18" /></span
          ><span class="project-name"
            >{{ project?.display_name || "连接本地工程…"
            }}<small>{{
              project?.product
                ? `${project.product} ${project.version ?? ""}`
                : "本机工作区"
            }}</small></span
          ><AppIcon name="chevron" :size="14" />
        </summary>
        <div class="project-detail">
          <label v-if="resources.projects.length > 1"
            >选择工程<select
              v-model="projectId"
              :disabled="task.active"
              @change="newConversation"
            >
              <option
                v-for="item in resources.projects"
                :key="item.id"
                :value="item.id"
              >
                {{ item.display_name }}
              </option>
            </select></label
          ><span>工作目录</span><code>{{ project?.path }}</code
          ><span>模型</span><code>{{ project?.model }}</code>
        </div>
      </details>
      <div class="sidebar-section-label conversations-label">
        会话记录 <span>{{ conversations.length }}</span>
      </div>
      <label class="conversation-search"
        ><AppIcon name="search" :size="14" /><input
          v-model="search"
          type="search"
          placeholder="查找会话"
          aria-label="查找会话"
      /></label>
      <nav class="conversation-list" aria-label="会话记录">
        <button
          v-for="item in conversations"
          :key="item.id"
          :class="[
            'conversation-link',
            { selected: item.id === conversationId },
          ]"
          :disabled="task.active || submitting"
          :aria-current="item.id === conversationId ? 'page' : undefined"
          :title="item.title"
          @click="selectConversation(item.id)"
        >
          <AppIcon name="chat" :size="16" /><span>{{ item.title }}</span
          ><time :datetime="item.created_at">{{
            shortDate(item.created_at)
          }}</time>
        </button>
        <p v-if="!conversations.length" class="sidebar-empty">
          {{ search ? "没有匹配的会话" : "开始对话后，会话会保存在这里。" }}
        </p>
      </nav>
      <footer class="sidebar-footer">
        <span class="connection-dot" :class="{ connected: !!project }"></span>
        <div>
          <strong>{{
            project
              ? "本地工程已连接"
              : resources.loading
                ? "正在连接…"
                : "工程未连接"
          }}</strong
          ><small>{{ project?.model ?? "检查本地服务是否启动" }}</small>
        </div>
        <AppIcon name="chip" :size="17" />
      </footer>
    </aside>
    <main class="chat-workspace">
      <header class="workspace-header">
        <button
          class="icon-button mobile-navigation"
          aria-label="打开会话导航"
          @click="sidebarOpen = true"
        >
          <AppIcon name="menu" />
        </button>
        <div class="workspace-breadcrumb">
          <AppIcon name="folder" :size="16" /><span>{{
            project?.display_name || "本地工程"
          }}</span
          ><span class="breadcrumb-separator">/</span>
          <h1>{{ conversation?.title || "新会话" }}</h1>
        </div>
        <button
          :class="['review-toggle', { selected: inspectorOpen }]"
          aria-label="工程审阅"
          :aria-expanded="inspectorOpen"
          aria-controls="delivery-panel"
          @click="inspectorOpen = !inspectorOpen"
        >
          <AppIcon name="panel" :size="16" /><span>工程审阅</span
          ><span
            v-if="task.delivery?.changed_files.length"
            class="change-count"
            >{{ task.delivery.changed_files.length }}</span
          >
        </button>
      </header>
      <div ref="transcript" class="transcript" @scroll="trackScroll">
        <div v-if="loadingHistory" class="loading-history">
          <span class="activity-dot"></span>正在读取会话…
        </div>
        <div v-else-if="!hasConversation" class="welcome">
          <span class="welcome-mark"><AppIcon name="chip" :size="31" /></span>
          <div class="welcome-eyebrow">
            {{ project?.product || "EMBEDDED WORKSPACE" }}
          </div>
          <h2>从一个模块，开始构建。</h2>
          <p>
            描述开发需求，让助手结合工程与规格书完成实现。<br />代码改动、原文依据和检查结果，都留在当前工作区。
          </p>
          <div v-if="resources.skills.length" class="quick-starts">
            <button
              v-for="item in resources.skills"
              :key="item.name"
              :class="{ selected: selectedSkill === item.name }"
              :aria-pressed="selectedSkill === item.name"
              @click="chooseSkill(item.name)"
            >
              <AppIcon name="book" :size="20" /><strong>{{
                item.title
              }}</strong
              ><span>{{ item.description }}</span
              ><AppIcon name="arrow" :size="15" class="quick-arrow" />
            </button>
          </div>
          <div v-if="project" class="welcome-context">
            <AppIcon name="folder" :size="14" /><span>{{
              project.display_name
            }}</span
            ><span v-if="project.product"
              >{{ project.product }} · {{ project.version }}</span
            >
          </div>
        </div>
        <div v-else class="message-list">
          <template v-for="item in earlier" :key="item.id">
            <article class="message user-message">
              <div class="user-bubble">
                <span v-if="item.skill" class="message-skill"
                  ><AppIcon name="book" :size="12" />{{ item.skill_title || item.skill }}</span
                >{{ item.prompt }}
              </div>
            </article>
            <article class="message assistant-message">
              <div class="assistant-identity">
                <span class="assistant-mark"
                  ><AppIcon name="chip" :size="16" /></span
                ><strong>助手</strong
                ><span>{{ statusLabels[item.status] }}</span>
              </div>
              <HistoryTrace :task-id="item.id" /><MarkdownContent
                :content="
                  item.response?.answer ||
                  item.response?.error ||
                  statusLabels[item.status]
                "
              />
            </article>
          </template>
          <article v-if="currentPrompt" class="message user-message">
            <div class="user-bubble">
              <span v-if="task.skillTitle" class="message-skill"
                ><AppIcon name="book" :size="12" />{{ task.skillTitle }}</span
              >{{ currentPrompt }}
            </div>
          </article>
          <article
            v-if="currentPrompt"
            class="message assistant-message current-message"
          >
            <div class="assistant-identity">
              <span class="assistant-mark"
                ><AppIcon name="chip" :size="16" /></span
              ><strong>助手</strong
              ><span v-if="task.active" class="assistant-working"
                >正在处理<span class="typing-dots">…</span></span
              >
            </div>
            <ExecutionTrace :events="task.events" :active="task.active" />
            <div v-if="task.answer" data-testid="answer">
              <MarkdownContent :content="task.answer" />
            </div>
            <p v-else-if="task.active" class="waiting-answer">
              <span class="activity-dot"></span>{{ activityLabel }}
            </p>
            <p v-else-if="!task.error" class="muted">本轮没有生成答复。</p>
            <div v-if="!task.active && task.answer" class="answer-actions">
              <button
                class="icon-button"
                :aria-label="copied ? '答复已复制' : '复制答复'"
                :title="copied ? '已复制' : '复制 Markdown'"
                @click="copyAnswer"
              >
                <AppIcon :name="copied ? 'check' : 'copy'" :size="15" /></button
              ><span v-if="task.result?.steps">{{ task.result.steps }} 步</span
              ><span v-if="task.result?.usage"
                >{{
                  task.result.usage.total_tokens.toLocaleString()
                }}
                tokens</span
              ><button
                v-if="task.delivery"
                class="delivery-link"
                @click="
                  reviewTab = task.delivery.decisions?.length
                    ? 'implementation'
                    : task.delivery.changed_files.length
                      ? 'changes'
                      : 'checks';
                  inspectorOpen = true;
                "
              >
                <AppIcon name="file" :size="14" />{{
                  task.delivery.changed_files.length
                    ? `${task.delivery.changed_files.length} 个文件改动`
                    : "查看执行记录"
                }}<AppIcon name="arrow" :size="12" />
              </button>
            </div>
            <button
              v-if="task.result?.verification && !task.active"
              :class="['verification-note', task.result.verification.status]"
              @click="
                reviewTab = 'checks';
                inspectorOpen = true;
              "
            >
              <AppIcon
                :name="
                  task.result.verification.status === 'passed'
                    ? 'check'
                    : 'clock'
                "
                :size="14"
              />{{ verificationLabel(task.result.verification.status)
              }}<AppIcon name="chevron" :size="12" />
            </button>
          </article>
        </div>
      </div>
      <div class="composer-area">
        <div v-if="formError || task.error" class="error-notice" role="alert">
          <AppIcon name="warning" :size="16" /><span>{{
            formError || task.error
          }}</span>
        </div>
        <div v-if="task.active" class="activity-bar" role="status">
          <span class="activity-dot"></span>{{ activityLabel
          }}<span class="activity-steps">{{ tools.length }} 项操作</span>
        </div>
        <form class="composer" @submit.prevent="start">
          <label class="sr-only" for="task-prompt">需求或后续修改</label
          ><textarea
            id="task-prompt"
            ref="input"
            v-model="prompt"
            rows="2"
            placeholder="描述需求，或继续提出修改…"
            :disabled="submitting || loadingHistory"
            @input="resizeInput"
            @keydown="onKeydown"
          ></textarea>
          <div class="composer-toolbar">
            <div class="composer-meta">
              <div
                v-if="resources.skills.length"
                :class="['skill-picker', { selected: selectedSkill }]"
              >
                <AppIcon name="book" :size="13" />
                <select
                  v-model="selectedSkill"
                  aria-label="选择开发技能"
                  :disabled="task.active || submitting"
                >
                  <option value="">自动选择技能</option>
                  <option v-for="item in resources.skills" :key="item.name" :value="item.name">
                    {{ item.title }}
                  </option>
                </select>
                <button
                  v-if="selectedSkill"
                  type="button"
                  aria-label="取消技能选择"
                  :disabled="task.active || submitting"
                  @click="selectedSkill = ''"
                ><AppIcon name="close" :size="12" /></button>
              </div>
              <div class="composer-context">
                <AppIcon name="folder" :size="14" /><span>{{
                  project?.display_name || "本地工作区"
                }}</span
                ><span class="composer-model">{{ project?.model }}</span>
              </div>
            </div>
            <button
              v-if="task.active"
              class="stop-button"
              type="button"
              :disabled="task.state === 'cancelling'"
              aria-label="取消任务"
              @click="cancel"
            >
              <AppIcon name="stop" :size="12" />{{
                task.state === "cancelling" ? "正在取消" : "停止"
              }}</button
            ><button
              v-else
              class="send-button"
              type="submit"
              :disabled="!canSend"
              aria-label="开始执行"
            >
              <AppIcon name="send" :size="18" />
            </button>
          </div>
        </form>
        <div class="composer-caption">
          <span :class="['task-status', task.state]" data-testid="task-status"
            ><span class="status-dot"></span
            >{{ statusLabels[task.state] }}</span
          ><span
            >Enter 发送 <span class="caption-divider">·</span> Shift + Enter
            换行</span
          >
        </div>
      </div>
    </main>
    <button
      v-if="inspectorOpen"
      class="inspector-scrim"
      aria-label="关闭工程审阅"
      @click="inspectorOpen = false"
    ></button
    ><DeliveryPanel
      v-if="inspectorOpen"
      id="delivery-panel"
      v-model:tab="reviewTab"
      :delivery="task.delivery"
      :result="task.result"
      :rechecking="task.recheckingTaskId === task.taskId && task.taskId !== null"
      :recheck-disabled="task.active || task.recheckingTaskId !== null"
      :recheck-error="task.recheckError"
      @recheck="task.recheck()"
      @close="inspectorOpen = false"
    />
  </div>
</template>
