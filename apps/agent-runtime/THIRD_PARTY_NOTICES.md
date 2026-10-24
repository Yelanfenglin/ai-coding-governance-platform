DSH goal, goal-round-driver, tool-goal, tool-todo and subagent lifecycle portions are adapted from
DeepSeek DSH commit `aa8262ec091698bae9a6b04773a6b5b06ad4aef2`.
Original paths are under `packages/goal/`, `packages/todo/` and `packages/subagent/`.
The model-facing goal policy, round prompt, closing instructions and tool
descriptions retain the original English text.

Workspace state ownership also references Pi's cwd-scoped session directory at
commit `cae9fd28a2d3d515c83ea495093e3768cc8efab6`. Shared nested usage references
OpenAI Agents SDK `Agent.as_tool` at commit `841dfdd938bfd4ca052475eb76d6245a356868e0`.
These references do not introduce either runtime as a dependency.

On-demand skill discovery references Pi's `packages/coding-agent/src/core/skills.ts`
(`loadSkillFromFile` and `formatSkillsForPrompt`) at the same Pi commit. Explicit
selection also references `AgentSession._expandSkillCommand` in `agent-session.ts`:
expand the body into the user message before execution. The local adapter uses
`--skill` and a Web task field instead of slash-command parsing; automatic discovery
uses `read_skill`. Both paths use the existing Agent loop and tools.

MIT License

Copyright (c) 2026 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
