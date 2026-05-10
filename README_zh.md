以下是该项目的中文翻译：

---

# MemGPT-mini

这是一个针对 MemGPT 论文核心理念的极简原生实现，专为学习而设计。代码量约为 500 行 Python，并刻意跳过了 Letta 在其基础上增加的生产级脚手架（如身份验证、多租户、追踪、沙箱、服务商适配及数据库迁移）。

当论文内容与 Letta 的实现不一致时，本仓库遵循 **Letta** 的逻辑。

## 架构

```
┌─────────────────────────────────────────────────┐
│ 主上下文 (每次迭代发送给 LLM)                      │
│ ┌─────────────────────────────────────────────┐ │
│ │ 系统提示词 + 此处渲染的 <core_memory>          │ │← 核心内存 (核心上下文)
│ └─────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────┐ │
│ │ 上下文内消息 (FIFO 队列)                       │ │
│ │   [可选的总结，角色为 user]                    │ │← 由压缩(Compaction)注入
│ │   用户 / 助手 / 工具 ...                       │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
      ▲                              ▲
      │ core_memory_append/replace   │ archival_memory_insert/search
      │                              │ conversation_search
┌─────┴──────────┐            ┌──────┴────────────────────┐
│ 核心内存 (RAM)  │            │ 存档内存 (pgvector)        │
│ (CoreMemory)   │            │ 召回内存 (Postgres 消息表)  │
└────────────────┘            └───────────────────────────┘

```

## 环境搭建

**前置条件：** Python 3.11+、带有 `pgvector` 扩展的 PostgreSQL，以及可用的 OpenAI 兼容 API Key。

1. 先创建本地环境变量文件。

```bash
cd /data/memgpt-mini
cp .env.example .env
```

至少需要配置：

- `DEEPSEEK_API_KEY`（或 `MEMGPT_CHAT_API_KEY`）
- `MEMGPT_PG_URI`
- 如果你希望存档记忆使用向量检索，再配置 `DASHSCOPE_API_KEY`

如果不配置 embedding key，存档记忆会回退为普通的 Postgres 文本搜索。

2. 启动带 `pgvector` 扩展的 PostgreSQL。

方案 A：本地 PostgreSQL

```bash
createdb memgpt_mini    # 如果尚未创建
psql memgpt_mini -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

方案 B：Docker

```bash
docker run -d --name memgpt-pg \
  -e POSTGRES_USER=memgpt \
  -e POSTGRES_PASSWORD=memgpt \
  -e POSTGRES_DB=memgpt_mini \
  -p 5432:5432 \
  pgvector/pgvector:pg16

docker exec -it memgpt-pg psql -U memgpt -d memgpt_mini \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

默认的 `.env.example` 已经指向上面这套 Docker 数据库：

```bash
MEMGPT_PG_URI=postgresql+asyncpg://memgpt:memgpt@localhost:5432/memgpt_mini
```

3. 安装依赖并运行 demo。

```bash
uv sync                 # 或者: pip install -e .
python demo.py
```

## 术语映射表

| 论文术语 | Letta 名称 | 本仓库对应位置 |
| --- | --- | --- |
| **系统指令 (System instructions)** | 系统提示词 (System prompt) | `agent.py` 中的 `DEFAULT_SYSTEM_PROMPT` |
| **工作上下文 (Working context)** | 核心内存块 (Core memory blocks) | `memory/core_memory.py` 中的 `Block`, `CoreMemory` |
| **FIFO 队列** | 上下文消息 (`in_context_messages`) | `MemGPTAgent.in_context` |
| **队列清理 / 递归总结** | 压缩 (Compaction) | `compaction.py` 中的 `compact()` |
| **存档存储 (Archival storage)** | 存档内存 (Archival memory) | `memory/archival.py` (基于 pgvector) |
| **召回存储 (Recall storage)** | 召回内存 (Recall memory) | `memory/recall.py` (消息表 + ILIKE 模糊查询) |
| **6 种内存工具** | 基础工具 (Base tools) | `tools/base_tools.py` |
| **函数链 (Function chaining)** | 工具循环 (Tool loop) | `MemGPTAgent.step()` 循环执行直至调用 `send_message` |

## 与论文的区别 (遵循 Letta 规范)

* **无警告阈值：** 论文在内存占用约 70% 时会注入“内存压力警告”，并期望 LLM 主动进行存档。Letta 移除了这一机制——压缩仅在达到清理阈值（0.9）时被动触发。本项目与 Letta 保持一致。
* **无 `request_heartbeat` 标志：** Letta V3 将心跳链替换为“只要有工具调用就进入循环”。本项目匹配 Letta V3 的逻辑。
* **压缩策略：** 论文会驱逐 50% 的消息并存储一个递归总结。Letta（以及本项目）保留最后 `message_buffer_min` 条原始消息，并将它们与系统提示词之间的所有内容进行总结。

## 文件布局

```text
memgpt/
  config.py              环境变量 → Config 数据类
  db.py                  SQLAlchemy 异步引擎 + Passage/MessageRow ORM 对象
  memory/
    core_memory.py       内存块 (Block) + 核心内存 (CoreMemory)
    archival.py          嵌入 (Embedding) + pgvector 搜索
    recall.py            消息持久化 + ILIKE 搜索
  tools/base_tools.py    6 种工具定义 + 分发逻辑 (dispatch)
  compaction.py          FIFO 总结策略 ([系统提示词, 总结, 尾部消息])
  agent.py               单步循环、工具分发、压缩触发器
demo.py                  交互式命令行界面 (CLI)

```
