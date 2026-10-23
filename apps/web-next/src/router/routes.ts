import type { RouteRecordRaw } from "vue-router";

export const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: "workspace",
    component: () => import("../views/AgentWorkspace.vue"),
  },
];
