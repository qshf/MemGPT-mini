# MemGPT-mini

A minimal from-scratch reproduction of the core ideas from the MemGPT paper, written for learning. It is ~500 lines of Python and intentionally skips the production scaffolding (auth, multi-tenant, tracing, sandboxing, providers, migrations) that Letta adds on top.

When the paper and Letta disagree, this repo follows **Letta**.

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Main context (sent to LLM every turn)           │
│ ┌─────────────────────────────────────────────┐ │
│ │ system prompt + <core_memory> rendered here │ │← CoreMemory (working context)
│ └─────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────┐ │
│ │ in_context messages (FIFO queue)            │ │
│ │   [optional summary as role=user]           │ │← injected by compaction
│ │   user / assistant / tool ...               │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
      ▲                              ▲
      │ core_memory_append/replace   │ archival_memory_insert/search
      │                              │ conversation_search
┌─────┴──────────┐            ┌──────┴────────────────────┐
│ CoreMemory     │            │ Archival (pgvector)       │
│ (RAM, blocks)  │            │ Recall (Postgres messages)│
└────────────────┘            └───────────────────────────┘
```

## Setup

Prereqs: Python 3.11+, PostgreSQL with the `pgvector` extension.

**Providers used by default:**
- Chat: DeepSeek (`deepseek-v4-pro`) via OpenAI-compatible API at `api.deepseek.com`, with `reasoning_effort=high` and thinking enabled.
- Embeddings: Alibaba DashScope (`text-embedding-v3`, 1024-dim) via OpenAI-compatible API at `dashscope.aliyuncs.com`.

Both are OpenAI-compatible, so the same `AsyncOpenAI` client handles them via `base_url` + `api_key`.

```bash
cd /data/memgpt-mini
cp .env.example .env    # keys are prefilled; edit if you want different ones
uv sync                 # or: pip install -e .
createdb memgpt_mini
psql memgpt_mini -c "CREATE EXTENSION IF NOT EXISTS vector;"
python demo.py
```

## What maps to what

| Paper term              | Letta name               | This repo                                                          |
| ----------------------- | ------------------------ | ------------------------------------------------------------------ |
| System instructions     | system prompt            | `agent.py` `DEFAULT_SYSTEM_PROMPT`                                 |
| Working context         | Core memory blocks       | `memory/core_memory.py` `Block`, `CoreMemory`                      |
| FIFO queue              | `in_context_messages`    | `MemGPTAgent.in_context`                                           |
| Queue flush / recursive summary | Compaction        | `compaction.py` `compact()`                                        |
| Archival storage        | Archival memory          | `memory/archival.py` (pgvector)                                    |
| Recall storage          | Recall memory            | `memory/recall.py` (messages table + ILIKE)                        |
| 6 memory tools          | Base tools               | `tools/base_tools.py`                                              |
| Function chaining       | Tool loop                | `MemGPTAgent.step()` loops until `send_message` is called          |

## Differences from the paper (follows Letta)

- **No warning threshold.** The paper injects a "memory pressure warning" at ~70% and expects the LLM to proactively archive. Letta removed this — compaction is purely reactive at the flush threshold (0.9). Mini matches Letta.
- **No `request_heartbeat` flag.** Letta V3 replaced heartbeat chaining with "loop whenever there's a tool call." Mini matches Letta V3.
- **Compaction strategy.** The paper evicts 50% and stores a recursive summary. Letta (and mini) keep the last `message_buffer_min` messages verbatim and summarize everything between them and the system prompt.

## Layout

```
memgpt/
  config.py              env vars → Config dataclass
  db.py                  SQLAlchemy async engine + Passage/MessageRow ORM
  memory/
    core_memory.py       Block + CoreMemory (working context)
    archival.py          embedding + pgvector search
    recall.py            message persistence + ILIKE search
  tools/base_tools.py    6 tool schemas + dispatch()
  compaction.py          FIFO summarization ([system, summary, tail])
  agent.py               step loop, tool dispatch, compaction triggers
demo.py                  interactive CLI
```
