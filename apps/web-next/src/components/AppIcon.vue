<script setup lang="ts">
withDefaults(defineProps<{ name: string; size?: number }>(), { size: 18 });
const paths: Record<string, string> = {
  chip: "M8 3v3m4-3v3m4-3v3M8 18v3m4-3v3m4-3v3M3 8h3m-3 4h3m-3 4h3m12-8h3m-3 4h3m-3 4h3M6 6h12v12H6zM9 9h6v6H9z",
  plus: "M12 5v14M5 12h14",
  chat: "M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 3V6a2 2 0 0 1 2-2Z",
  folder: "M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v10H3Z",
  chevron: "m9 5 7 7-7 7",
  panel: "M3 4h18v16H3zM15 4v16",
  file: "M6 3h8l4 4v14H6zM14 3v5h4M9 12h6m-6 4h6",
  code: "m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 16",
  book: "M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Zm0 0v15",
  check: "m5 12 4 4L19 6",
  terminal: "m4 6 6 6-6 6m9 0h7",
  settings: "M4 7h10m4 0h2M4 17h2m4 0h10M14 4v6M10 14v6",
  send: "M12 19V5m-6 6 6-6 6 6",
  stop: "M6 6h12v12H6z",
  close: "m6 6 12 12M18 6 6 18",
  menu: "M4 6h16M4 12h16M4 18h16",
  copy: "M8 8h12v12H8zM4 15V4h11",
  search: "M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0Zm-2 5 6 6",
  clock: "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-5v6l4 2",
  arrow: "M5 12h14m-6-6 6 6-6 6",
  warning: "M12 3 2 21h20L12 3Zm0 6v5m0 3v1",
};
</script>

<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.65"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path :d="paths[name] || paths.file" />
  </svg>
</template>
