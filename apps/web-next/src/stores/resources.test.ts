import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useResourceStore } from "./resources";

describe("local resources", () => {
  beforeEach(() => { setActivePinia(createPinia()); vi.unstubAllGlobals(); });

  it("loads the local project and persisted conversations", async () => {
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify([{ id: "local", path: "project" }])))
      .mockResolvedValueOnce(new Response(JSON.stringify([{ id: "old", title: "Existing" }])))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "new", title: "New" }), { status: 201 }));
    vi.stubGlobal("fetch", fetch);
    const resources = useResourceStore();
    await resources.load();
    await resources.createConversation("New", "local");
    expect(resources.projects[0]?.id).toBe("local");
    expect(resources.conversations.map((item) => item.id)).toEqual(["new", "old"]);
    expect(fetch.mock.calls[0]![0]).toBe("/api/agent/projects");
    expect(JSON.parse(fetch.mock.calls[2]![1].body)).toEqual({ title: "New", project_id: "local" });
  });
});
