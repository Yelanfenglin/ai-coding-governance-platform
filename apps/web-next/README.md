# 工程工作台

Vue 3、TypeScript 与 Pinia 构建的本地 Agent 工作区。左侧管理工程与会话，中间展示回答、思考摘要和工具活动，右侧审阅实现依据、文件差异、规格证据与检查结果。

## 运行

按 [项目说明](../../README.md) 配置 Agent 与 RAG，从仓库根目录 `npm run dev`，打开 `http://127.0.0.1:5174`。构建后 `npm start`，由本地 Agent 在 8081 提供页面与 API。

开发代理仅将 `/api/agent` 转发至本机服务。会话续接、取消和事件重放使用同一 API，最终任务状态包含宿主检查结果；模型文本以 Markdown 渲染并关闭原生 HTML。

## 检查

```powershell
npm test
npm run build
npm run test:e2e
```

E2E 使用隔离工程及本地脚本 HTTP 模型/RAG 对端，需要 Agent 的 `.[web,dev]` 环境与 Chrome，不消耗付费模型 token。

工程与凭据由本机启动配置选择；本模块不接收任意文件系统根目录或模型密钥。接口装配见 `apps/agent-runtime/app/web.py`。
