# 运行与验证

[首页](../README.md) · [原理](architecture.md) · [监控](observability.md)

## 启动顺序

**准备环境 → 配置模型和知识库 → 启动 RAG → 启动工作台**

需要 Python 3.12+、Node.js 22+、PowerShell 7；知识库使用 PostgreSQL 和 Milvus。

<details>
<summary>首次安装与服务启动命令</summary>

从仓库根目录执行：

```powershell
$env:PYTHONNOUSERSITE='1'
npm install
py -3.12 -m venv apps/agent-runtime/.venv
& apps/agent-runtime/.venv/Scripts/python.exe -m pip install -e "./apps/agent-runtime[web,dev]"
py -3.12 -m venv apps/rag-api/.venv
& apps/rag-api/.venv/Scripts/python.exe -m pip install -e "./apps/rag-api[dev]"
Copy-Item apps/agent-runtime/.env.example apps/agent-runtime/.env
Copy-Item apps/rag-api/.env.example apps/rag-api/.env
```

填写 [Agent 环境变量](../apps/agent-runtime/.env.example)中的工程目录与模型凭据，以及 [RAG 环境变量](../apps/rag-api/.env.example)中的数据库、向量库和 provider。向量模型、维度与 collection 保持一致。

在一个终端启动知识库服务：

```powershell
docker compose -f apps/rag-api/infra/milvus/compose.yaml up -d
Push-Location apps/rag-api
& .venv/Scripts/python.exe -m alembic upgrade head
& .venv/Scripts/python.exe -B scripts/serve.py
```

在另一个终端回到仓库根目录，执行下方启动命令。原始 PDF 和知识库数据由部署者准备；解析、入库分别使用 `python -m app.parser.parse --help`、`python -m app.ingest --help`，需在 RAG 项目环境中执行。

</details>

```powershell
npm run dev
```

开发入口：`http://127.0.0.1:5174`。构建后执行 `npm run build`、`npm start`，入口为 `http://127.0.0.1:8081`。

## 日常检查

```powershell
npm test
npm run build
npm run test:e2e
```

Python 检查使用各模块虚拟环境并设置 `PYTHONNOUSERSITE=1`；浏览器 E2E 使用隔离工程和脚本对端。

<details>
<summary>固定工程任务、验收与 ADC 反例</summary>

用例在 [Agent evaluation](../apps/agent-runtime/evaluation/cases/)，将对应 [fixture](../apps/agent-runtime/evaluation/fixtures/) 复制为独立工程后运行。主机行为检查需要 GCC，可通过 `CC` 指定编译器。

宿主通过 `--verification <配置.json>` 提供检查命令；开发中返回反馈，交付时复验最终版本。配置示例：

```json
{"command": ["{python}", "tests/check_firmware.py"], "cwd": "{workspace}", "timeout": 60}
```

在 `apps/agent-runtime` 的项目环境中，离线检查与评分：

```powershell
.venv/Scripts/python.exe -m evaluation.runner check --case evaluation/cases/multi_module_change.json --workspace <工程目录> --out <检查目录>
.venv/Scripts/python.exe -m evaluation.runner score --case evaluation/cases/multi_module_change.json --delivery <delivery.json> --workspace <工程目录> --out <评分目录>
```

ADC 迁移用例为 `chip_migration_adc.json`；保留 fixture 的 `source/target` 结构。安装 `.[evaluation]` 后开启序列探索，修复后可重放反例：

```powershell
.venv/Scripts/python.exe evaluation/checks/adc_migration.py --workspace <工程目录> --out <工程外目录> --explore-sequences --sequence-examples 300 --sequence-steps 30
.venv/Scripts/python.exe evaluation/checks/adc_sequences.py --workspace <工程目录> --out <工程外目录> --replay <counterexample.json>
```

检查产物写到状态目录或工程外，历史评分使用对应轮次的最终工程版本。

</details>

<details>
<summary>交付后复验与撤回</summary>

Web 的“工程审阅”提供重新验收和撤回入口。CLI 在 `apps/agent-runtime` 的项目环境执行：

```powershell
.venv/Scripts/python.exe -m app.reverification --workspace <工程目录> --delivery <delivery.json> --verification <配置.json>
.venv/Scripts/python.exe -m app.undo --workspace <工程目录> --delivery <delivery.json>
```

复验核对工程与宿主配置；撤回核对全部受影响文件。Web 在这些操作期间互斥执行，CLI 在工程空闲时使用。结果单独保存，原任务、diff 与验收记录继续保留；工程外命令副作用单独处理。

</details>

<details>
<summary>RAG 接口与离线评分</summary>

默认前缀 `/api/rag`。设置 `RAG_API_KEY` 后携带 `X-API-Key`；Agent 的 `RAG_API_URL` 包含完整前缀。

| 接口 | 用途 |
|---|---|
| `GET /health` | 知识库就绪状态 |
| `GET /search` | 按问题、型号、版本检索 |
| `GET /documents/{doc_id}/context/{chunk_id}` | 展开原文 |
| `POST /query` | 检索、生成与引用检查 |

`rerank=true` 需服务端启用重排；`candidate_k` 控制候选池，`top_k` 控制返回量。未配置返回 422，重排降级保留为空的重排分数与状态。

在 `apps/rag-api` 使用项目环境：

```powershell
$env:PYTHONNOUSERSITE='1'
.venv/Scripts/python.exe -m pytest -q -m "not integration"
.venv/Scripts/python.exe -B evaluation/runner.py validate --pdf-root <原始PDF目录>
.venv/Scripts/python.exe -B evaluation/runner.py score --run-dir <已有检索产物> --split development
```

[retrieval](../apps/rag-api/evaluation/retrieval/) 保存冻结题库、来源映射与评分程序。题库文件保留构建时记录；已有实测结果见[首页](../README.md#效果与验证)。评分只读取已有产物，原文校验需要本机 PDF 与 chunk 文件。

</details>
