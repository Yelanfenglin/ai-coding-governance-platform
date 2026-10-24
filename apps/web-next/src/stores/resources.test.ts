import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useResourceStore } from "./resources";

describe("local resources", () => {
  beforeEach(() => { setActivePinia(createPinia()); vi.unstubAllGlobals(); });

  it("loads the local project and persisted conversations", async () => {
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify([{ id: "local", path: "project" }])))
      .mockResolvedValueOnce(new Response(JSON.stringify([{ id: "old", title: "Existing" }])))
      .mockResolvedValueOnce(new Response(JSON.stringify([{ name: "module-development", title: "模块开发", description: "实现模块" }])))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "new", title: "New" }), { status: 201 }));
    vi.stubGlobal("fetch", fetch);
    const resources = useResourceStore();
    await resources.load();
    await resources.createConversation("New", "local");
    expect(resources.projects[0]?.id).toBe("local");
    expect(resources.skills[0]?.name).toBe("module-development");
    expect(resources.conversations.map((item) => item.id)).toEqual(["new", "old"]);
    expect(fetch.mock.calls[0]![0]).toBe("/api/agent/projects");
    expect(fetch.mock.calls[2]![0]).toBe("/api/agent/skills");
    expect(JSON.parse(fetch.mock.calls[3]![1].body)).toEqual({ title: "New", project_id: "local" });
  });
});
