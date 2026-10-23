import { defineStore } from "pinia";
import { apiRequest } from "../api";
import type { TaskRecord } from "./task";

export interface ProjectRecord {
  id: string;
  display_name: string;
  path: string;
  model: string;
  product: string | null;
  version: string | null;
}

export interface ConversationRecord {
  id: string;
  project_id: string;
  title: string;
  created_at: string;
}

export const useResourceStore = defineStore("resources", {
  state: () => ({
    projects: [] as ProjectRecord[],
    conversations: [] as ConversationRecord[],
    loading: false,
  }),
  actions: {
    async load() {
      this.loading = true;
      try {
        [this.projects, this.conversations] = await Promise.all([
          apiRequest<ProjectRecord[]>("/projects"),
          apiRequest<ConversationRecord[]>("/conversations"),
        ]);
      } finally {
        this.loading = false;
      }
    },
    async createConversation(title: string, projectId: string) {
      const conversation = await apiRequest<ConversationRecord>("/conversations", {
        method: "POST", body: JSON.stringify({ title, project_id: projectId }),
      });
      this.conversations.unshift(conversation);
      return conversation;
    },
    async history(identifier: string) {
      return apiRequest<ConversationRecord & { tasks: TaskRecord[] }>(`/conversations/${identifier}`);
    },
  },
});
