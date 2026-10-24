import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

test("real HTTP runtime: reasoning, files, RAG, delivery, continuation, cancellation and responsive UI", async ({
  page,
  request,
}, testInfo) => {
  test.setTimeout(60000);
  await page.setViewportSize({ width: 1440, height: 940 });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const projects = await (await request.get("/api/agent/projects")).json();
  const workspace = projects[0].path;
  await page.goto("/");
  await expect(page.getByText("嵌入式编程助手", { exact: true })).toBeVisible();
  await expect(page.getByText("Codex", { exact: false })).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "从一个模块，开始构建。" }),
  ).toBeVisible();
  expect(
    (await (await request.get("/api/agent/conversations")).json()).length,
  ).toBe(0);
  await page.screenshot({ path: testInfo.outputPath("desktop-empty.png") });
  await page.getByRole("button", { name: "芯片改型" }).click();
  await expect(page.getByRole("combobox", { name: "选择开发技能" })).toHaveValue("chip-migration");
  await page.getByRole("button", { name: "取消技能选择" }).click();
  await expect(page.getByRole("combobox", { name: "选择开发技能" })).toHaveValue("");
  await page.getByRole("button", { name: "模块开发" }).click();
  await expect(page.getByRole("combobox", { name: "选择开发技能" })).toHaveValue("module-development");
  await expect(page.locator("textarea")).toHaveValue("");
  expect(
    (await (await request.get("/api/agent/conversations")).json()).length,
  ).toBe(0);
  await page
    .locator("textarea")
    .fill("读取手册和 seed.c，创建模块并检查；用户确认 HOST_ONLY 环境。");
  await page.locator("textarea").dispatchEvent("keydown", {
    key: "Enter",
    isComposing: true,
    keyCode: 229,
  });
  expect(
    (await (await request.get("/api/agent/conversations")).json()).length,
  ).toBe(0);
  await page.locator("textarea").press("Enter");
  await expect(page.getByTestId("task-status")).toHaveText("执行完成", {
    timeout: 20000,
  });
  await expect(page.getByTestId("answer")).toContainText("已创建 generated.c");
  await expect(page.locator(".user-message .message-skill")).toHaveText("模块开发");
  await expect(page.getByRole("combobox", { name: "选择开发技能" })).toHaveValue("");
  await expect(page.getByTestId("answer").locator("h2")).toHaveText("已完成");
  await expect(page.getByTestId("answer").locator("pre code")).toContainText(
    "int value = 2;",
  );
  await expect(page.locator(".reasoning-block")).toHaveCount(5);
  await expect(page.locator(".reasoning-block[open]")).toHaveCount(0);
  await page.locator(".reasoning-block > summary").first().click();
  await expect(page.getByTestId("reasoning").first()).toContainText(
    "先检查当前工程和规格依据",
  );
  await page.locator(".tool-group > summary").first().click();
  await expect(page.getByText("检索规格书", { exact: true })).toBeVisible();

  await page
    .getByRole("button", { name: "未配置独立验收", exact: true })
    .click();
  await expect(page.getByTestId("verification")).toContainText(
    "未配置独立验收",
  );
  await page.getByRole("tab", { name: /实现/ }).click();
  const decision = page.getByTestId("implementation-decision");
  await expect(decision).toHaveCount(1);
  await expect(decision).toContainText("模块示例的配置与接入");
  await expect(decision).toContainText("generated.c:1–1");
  await expect(decision).toContainText("位置已核对");
  await expect(decision).toContainText("执行通过 · 版本一致");
  await expect(decision).toContainText("未做厂商编译或板卡验证");
  await page.screenshot({
    path: testInfo.outputPath("desktop-implementation.png"),
  });
  await page.getByRole("button", { name: "查看文件差异" }).click();
  await expect(page.getByTestId("diff")).toContainText("+int value = 2;");
  expect(await readFile(join(workspace, "generated.c"), "utf8")).toBe(
    "int value = 2;\n",
  );
  await page.getByRole("tab", { name: /实现/ }).click();
  await page.getByRole("button", { name: /M9F6820 V1.7 · p.113/ }).click();
  await expect(page.getByRole("tab", { name: /依据/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.locator(".evidence-item")).toHaveAttribute("open", "");
  await expect(page.locator(".evidence-text")).toContainText("测试手册");
  await expect(page.locator(".evidence-location")).toContainText("文档");
  await page.screenshot({ path: testInfo.outputPath("desktop-evidence.png") });
  await page.getByRole("tab", { name: /改动/ }).click();
  await page.screenshot({ path: testInfo.outputPath("desktop-delivery.png") });
  await page.getByRole("button", { name: "收起审阅面板" }).click();
  await page.reload();
  await expect(page.getByTestId("task-status")).toHaveText("执行完成");
  await expect(page.locator(".user-message .message-skill")).toHaveText("模块开发");
  await page.getByRole("button", { name: /1 个文件改动/ }).click();
  await expect(decision).toContainText("模块示例的配置与接入");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".sidebar")).not.toBeInViewport();
  await expect(decision).toBeVisible();
  for (const target of [
    decision,
    page.getByRole("button", { name: "收起审阅面板" }),
  ]) {
    const bounds = await target.boundingBox();
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  await page.screenshot({
    path: testInfo.outputPath("mobile-implementation.png"),
  });
  await page.getByRole("button", { name: "收起审阅面板" }).click();
  await page.setViewportSize({ width: 1440, height: 940 });

  await page
    .locator("textarea")
    .fill("继续把上一轮的 value 改成 3，保留原工程。");
  await page.getByRole("button", { name: "开始执行" }).click();
  await expect(page.getByTestId("answer")).toContainText("将 value 改为 3", {
    timeout: 20000,
  });
  await expect(page.getByTestId("task-status")).toHaveText("执行完成");
  expect(await readFile(join(workspace, "generated.c"), "utf8")).toBe(
    "int value = 3;\n",
  );
  await page.locator(".history-trace > summary").first().click();
  await expect(page.locator(".history-trace .reasoning-block")).toHaveCount(5);
  await page
    .locator(".history-trace .reasoning-block > summary")
    .first()
    .click();
  await expect(
    page.locator('.history-trace [data-testid="reasoning"]').first(),
  ).toContainText("先检查当前工程和规格依据");

  await page.locator("textarea").fill("等待取消");
  await page.getByRole("button", { name: "开始执行" }).click();
  await expect(page.getByTestId("answer")).toContainText("正在等待取消");
  await expect(
    page.locator(".current-message .reasoning-block[open]"),
  ).toHaveCount(1);
  await expect(
    page.locator('.current-message [data-testid="reasoning"]'),
  ).toContainText("先检查当前工程和规格依据");
  await page.getByRole("button", { name: "取消任务" }).click();
  await expect(page.getByTestId("task-status")).toHaveText("已取消", {
    timeout: 10000,
  });
  await page.reload();
  await expect(page.getByTestId("task-status")).toHaveText("已取消");
  await page.locator("textarea").fill("取消后继续");
  await page.getByRole("button", { name: "开始执行" }).click();
  await expect(page.getByTestId("answer")).toContainText("已在原会话续接");
  await expect(page.getByTestId("task-status")).toHaveText("执行完成");

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("combobox", { name: "选择开发技能" }).selectOption("chip-migration");
  await expect(page.locator("textarea")).toBeVisible();
  const composer = await page.locator(".composer").boundingBox();
  expect(composer!.y + composer!.height).toBeLessThan(844);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  await page.getByRole("button", { name: "打开会话导航" }).click();
  await expect(page.getByRole("button", { name: /新建会话/ })).toBeVisible();
  await page.getByRole("button", { name: "关闭会话导航" }).click();
  await page.getByRole("button", { name: "工程审阅", exact: true }).click();
  await expect(
    page.getByRole("complementary", { name: "工程审阅" }),
  ).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("mobile-review.png") });
  await page.getByRole("button", { name: "收起审阅面板" }).click();
  await page.screenshot({ path: testInfo.outputPath("mobile-chat.png") });

  const log = (
    await readFile(
      join(workspace, ".agent-runtime", "fixture-requests.jsonl"),
      "utf8",
    )
  )
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  const model = log.filter((row) => row.method === "POST");
  expect(model.every((row) => row.body.model === "deepseek-flash")).toBe(true);
  expect(
    model.every(
      (row) =>
        row.body.thinking.type === "enabled" &&
        row.body.reasoning_effort === "high",
    ),
  ).toBe(true);
  const firstUser = model[0].body.messages.find((item: { role: string }) => item.role === "user");
  expect(firstUser.content).toContain('<skill name="module-development"');
  expect(firstUser.content).toContain("# 工程内的可复用模块开发");
  expect(model[0].body.messages.filter((item: { role: string }) => item.role === "tool")).toHaveLength(0);
  expect(
    model
      .at(-1)
      .body.messages.some(
        (item: { role: string; content: string }) =>
          item.role === "assistant" &&
          item.content?.includes("将 value 改为 3"),
      ),
  ).toBe(true);
  expect(
    log.some((row) => row.method === "GET" && row.path.startsWith("/search?")),
  ).toBe(true);
  expect(
    log.some((row) => row.method === "GET" && row.path.includes("/context/")),
  ).toBe(true);
  expect(errors).toEqual([]);
});
