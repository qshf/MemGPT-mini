# 差异对比：MemGPT 论文 vs Letta vs memgpt-mini

三条时间线：

```
2023 MemGPT 论文 (arXiv:2310.08560) ── 2024-2025 Letta 工程化 ── 2026 memgpt-mini 学习版
```

本文档记录三层演化中每一步**为什么这么改**。当论文和 Letta 冲突时，mini 跟 Letta；当 Letta 过于复杂时，mini 在保留语义的前提下简化。

---

## Part 1: MemGPT 论文 → Letta

### 1.1 Main Context 结构：三段不变，但 Working Context 变多块

| | 论文 | Letta |
|---|---|---|
| System instructions | ✔ | ✔ |
| Working context | **单一**定长可读写区域 | **多个命名 Block**（`human`、`persona`、自定义 label），每个 Block 独立 `value` / `limit` / `description` |
| FIFO queue | ✔ | ✔ |

**为什么改**：单一 working context 在实践中很难让 LLM 正确使用——写 "persona" 和写 "human facts" 的策略不一样。拆成多 block 后，system prompt 里可以给每个 block 挂上 `description`，LLM 知道什么该写到哪。工具签名也从"改 working_context"变成"改 `label=human` 的 block"，更精准。

Letta 源码：[letta/schemas/block.py](/data/letta/letta/schemas/block.py)、[letta/services/block_manager.py](/data/letta/letta/services/block_manager.py)

### 1.2 Warning 阈值（70%）：完全删除

| | 论文 | Letta |
|---|---|---|
| 70% warning | 注入 `[SYSTEM] memory pressure` 消息，期望 LLM 主动归档 | ❌ 没有 |
| 100% flush | 强制驱逐 + 摘要 | 改为 90% trigger（仍是"强制"分支） |

**为什么删**：论文的设想是 LLM 收到 warning 就会主动调 `archival_storage.insert`。实际跑下来两个问题：

1. LLM 对 warning 的服从率不稳定（有时候忽略，有时候反而大段复述），行为不可预测。
2. 就算 LLM 配合，它归档的是**自己认为重要的内容**——经常漏掉真正的关键信息。

Letta 的结论：与其依赖 LLM 自主，不如让系统**强制在 90% 时摘要**，重要信息的归档改由用户在对话中直接要求（"请记住 X"→ LLM 调 `archival_memory_insert`）。这是个务实的妥协——把"何时归档"的决策权部分交还给用户，换取系统行为的确定性。

来源：[letta/services/summarizer/thresholds.py](/data/letta/letta/services/summarizer/thresholds.py) `SUMMARIZATION_TRIGGER_MULTIPLIER=0.9`，没有 warning 阈值。

### 1.3 压缩算法：从"驱逐 50% + 递归摘要"升级到 sliding_window

| | 论文 | Letta |
|---|---|---|
| 驱逐比例 | 固定 50% | 初始 30%，loop 每轮 +10% 直到 `count_tokens_after ≤ goal_tokens` |
| 摘要类型 | **递归摘要**（旧 summary + 新 evict → 合并生成新 summary） | **非递归**（每次对当前全部 evict 重新生成） |
| 切点位置 | 论文未明确 | **必须是 assistant 消息**（保证 tool_calls/tool_result 不被切开） |
| 兜底 | 无 | 压完若仍超阈值 → 自动切 `all` 模式再压一次 |

**为什么改**：

- **动态比例**：固定 50% 有时压不够（长摘要+长尾巴仍超），有时压过头（浪费）。动态 loop 直到"真的够"为止，省 token 省摘要质量。
- **非递归**：递归摘要会**逐轮衰减信息**——第 1 轮摘要"Alice 是 Go 开发者"，第 2 轮变成"用户在做技术工作"，第 3 轮可能只剩"早期对话涉及技术"。Letta 改成每次对全量 evict 重新生成，避免信息腐烂。代价是每次摘要的 prompt 更长，但摘要质量稳定。
- **assistant 切点**：这是**工程必须**，不是设计选择。OpenAI / Anthropic / DeepSeek 的 chat API 都要求 `role=tool` 必须紧跟 `assistant tool_calls`，切错位置直接 400。
- **兜底**：偶尔会遇到"摘要本身就很长"的病态情况，需要用更激进的 all 模式再来一次。

来源：[letta/services/summarizer/summarizer_sliding_window.py](/data/letta/letta/services/summarizer/summarizer_sliding_window.py)

### 1.4 `request_heartbeat` 机制：V3 删除

| | 论文 | Letta V3 |
|---|---|---|
| 函数链式调用 | 函数返回时带 `request_heartbeat=true` → 继续推理；`false` → yield 等外部事件 | ❌ 只要有 tool_call 就继续循环，不看任何标志 |

**为什么删**：`request_heartbeat` 让 LLM 承担"判断自己是否需要继续思考"的责任——这和 LLM 的现代用法不符。现在的做法是：

```
while last_response has tool_calls:
    execute_tools()
    call LLM again
```

循环自然终止于"LLM 不再调工具"，不需要 LLM 自己说"我要继续"。更简单、更不易出错。

Letta V3 的注释：`No heartbeats (loops happen on tool calls)`（[letta_agent_v3.py:107](/data/letta/letta/agents/letta_agent_v3.py)）。

### 1.5 Archival Storage：从"单一向量库"升级到"Archive 实体"

| | 论文 | Letta |
|---|---|---|
| 归属 | 跟 agent 绑定（论文隐含一对一） | `Archive` 是独立实体，通过 `archives_agents` 关联表挂给一个或多个 agent |
| 共享 | ❌ 不支持 | ✔ 多 agent 可共享同一 archive（团队知识库场景） |
| 后端 | PostgreSQL + pgvector + HNSW | 同 |

**为什么改**：团队场景下，多个 agent（比如"客服 agent"、"销售 agent"）可能需要共享同一份产品知识库。论文的"归 agent 私有"模型做不到这点。Letta 的 `Archive` 抽象让同一份 passage 可以被多个 agent 读写。

### 1.6 Recall Storage：从"分页搜索"升级到 Hybrid Search

| | 论文 | Letta |
|---|---|---|
| 接口 | 分页查询 | 全文 + 语义混合搜索、按 role / 时间窗过滤 |
| 后端 | 未明确 | Postgres messages 表 |

**为什么改**：单纯的分页搜索只能"按时间翻页"，查具体内容几乎没用。Letta 加了词汇匹配 + 向量相似度混排，以及 `roles=['user', 'assistant']` 这种过滤。

## Part 2: Letta → memgpt-mini

mini 的目标是"用最少代码讲清核心原理"。所以有些 Letta 的生产特性被砍了，但砍的每一刀都记在这里。

### 2.1 保留的核心（和 Letta 语义等价）

| 功能 | 实现位置 | 等价度 |
|---|---|---|
| 多 block 的 CoreMemory | [memory/core_memory.py](../memgpt/core_memory.py) | ✔ 完全等价 |
| Archival（pgvector） | [memory/archival.py](../memgpt/archival.py) | ✔ 完全等价，加了 `archive_id` 隔离 |
| Recall（messages 表） | [memory/recall.py](../memgpt/recall.py) | ✔ 按 `agent_id` 过滤，加了切词 OR + hit-count 排序 |
| sliding_window 压缩 | [compaction.py](../memgpt/compaction.py) | ✔ 算法 1:1 对齐 |
| post-compact fallback | [compaction.py `compact`](../memgpt/compaction.py) | ✔ 完全等价 |
| assistant 切点 | [compaction.py `_is_valid_cutoff`](../memgpt/compaction.py) | ✔ 完全等价 |
| 两个压缩触发入口 | [agent.py `_maybe_compact` / `BadRequestError`](../memgpt/agent.py) | ✔ 完全等价 |
| SLIDING_PROMPT / ALL_PROMPT | [compaction.py](../memgpt/compaction.py) | ✔ 从 Letta 翻译精简 |
| 6 个 base tools | [tools/base_tools.py](../memgpt/tools/base_tools.py) | ✔ 签名对齐 Letta |

### 2.2 简化的部分（等价但实现更薄）

**Archive 模型**

- Letta：`Archive` 实体 + `archives_agents` 关联表
- mini：`passages.archive_id` 单列字符串；`MemGPTAgent(archive_id=...)` 构造时传一个 id。两个 agent 想共享就传同一个 id。

放弃了"一个 agent 挂多个 archive"的场景（很少见），换来零关联表的简洁 ORM。

**`role=summary` 消息**

- Letta：内部 ORM 有 `MessageRole.summary`，提供给 provider 前 `to_openai_dict()` 转成 `role=user`
- mini：没有中间 ORM，压缩直接产出 `role=user`。语义等价（provider 看到的东西完全一样）。

**Provider 抽象**

- Letta：`llm_api/anthropic_client.py` / `openai_client.py` / `deepseek_client.py` 等多 provider 适配层
- mini：直接用 `AsyncOpenAI(base_url=...)`。DeepSeek/DashScope/OpenAI 都是 OpenAI-compatible，所以一份代码够了。

**工具 Sandbox**

- Letta：工具可以在 E2B / Modal / 本地 venv 里跑
- mini：工具直接 Python async 调用。mini 只有 6 个 base tool（内置的 core_memory / archival / recall 操作），没有"用户自定义代码"场景，沙箱没必要。

**Tracing / Telemetry / Temporal**

- Letta：`@trace_method` OTel 装饰器、Temporal workflow、metrics
- mini：全删。print 一行 `[compaction:...]` 就够看了。

**数据库迁移**

- Letta：Alembic migration 体系
- mini：`CREATE TABLE ... IF NOT EXISTS` + `ADD COLUMN IF NOT EXISTS`。schema 改动直接改 ORM 类，下次 `init_schema()` 自动补列。

### 2.3 增加的小改进（Letta 也有，但 mini 做得更彻底）

**DB 里显式存 `tool_name`**

Letta 的 `Message` ORM 有一串 tool 相关字段。mini 加了一个专门的 `messages.tool_name` 列，只在 `role=tool` 的行上填。作用：

- `conversation_search` 结果里能直接显示"是哪个工具返回的内容"，对调试非常有用
- 不进 LLM payload，纯本地元数据

**空 Block 渲染成自闭合 XML**

Letta 的渲染 `<block>\n\n</block>`（空体）。mini 改成 `<block ... empty="true"/>`，避免 LLM 把渲染器的占位符（例如先前版本的 `(empty)` 字面）误认成 block 内容，导致连锁错误。

**切词 OR + hit-count 排序（conversation_search）**

Letta 有 hybrid search，mini 简化成"按空格切词 → OR 匹配 → 按命中词数排序"。一眼能看懂，效果对学习足够。

### 2.4 保留简化的争议点

这些是 mini **明知不是 Letta 的做法**但选择保留的：

**`send_message` 不是强制**

- Letta：`send_message` 是 base tool，必须调用才能产生可见回复
- mini：如果 LLM 直接用 `choice.content` 回复（常见于 DeepSeek 无 tool 的简单对话），也接受。

理由：`send_message` 在论文里是 DM-style agent 的核心抽象，但 DeepSeek 经常不调用它直接 content 回复。强制会降低成功率，对学习没必要。

**Core memory 不持久化**

- Letta：`blocks` 表存每个 block 的 value
- mini：只在进程内存里。重启丢。

理由：要演示"session 内的记忆管理"不需要持久化；真要持久化加个 `blocks` 表即可，几十行代码。

## Part 3: 速查表

### 记忆三层：论文 / Letta / mini

| 层级 | 论文 | Letta | mini |
|---|---|---|---|
| **Core memory** | 单一 working context | 多 Block，`block` + `blocks_agents` 关联表，`BlockManager` | 单 `CoreMemory` 对象持有 `{label: Block}` dict，纯内存 |
| **Archival** | 向量库（pgvector） | `Archive` + `archives_agents` + `passages` + `PassageManager` | `passages` 表按 `archive_id` 列隔离，`ArchivalMemory(archive_id=...)` |
| **Recall** | 消息库 + 分页搜索 | `messages` 表 + hybrid search + `MessageManager` | `messages` 表按 `agent_id` 列隔离，切词 OR + hit-count 排序 |
| **Compaction** | 50% 固定 + 递归摘要 | sliding_window / all，动态比例 loop | 同 Letta，算法 1:1 |
| **Tools** | 7 个（含 read_memory 等） | 6 个 base tools + 用户自定义 | 6 个 base tools（对齐 Letta 签名） |

### 压缩：论文 / Letta / mini

| 环节 | 论文 | Letta | mini |
|---|---|---|---|
| warning 阈值 | 70% 注入系统消息 | 无 | 无 |
| flush 阈值 | 100% | 90% (`context_window * 0.9`) | 同 Letta |
| 驱逐比例 | 50% 固定 | 30% 初始 + loop | 同 Letta |
| 摘要方式 | 递归 | 非递归（全量重摘） | 同 Letta |
| 切点位置 | 未明确 | assistant 消息 | 同 Letta |
| 兜底 | 无 | sliding 不够 → all | 同 Letta |
| 摘要 role | 未明确 | `summary`（内部）→ `user`（provider） | 直接 `user` |

### 行为分层：论文 / Letta / mini

| 行为 | 论文 | Letta | mini |
|---|---|---|---|
| 函数链 | `request_heartbeat=true/false` | 只要有 tool_call 就继续循环（V3） | 同 Letta V3 |
| 事件驱动 | user / system / 定时 | `MessageCreate` 统一 + Temporal 定时 | 只 user（`MemGPTAgent.step(user_msg)`） |
| Sandbox | 未涉及 | E2B / Modal / 本地 venv | 无（直接 await） |
| Provider | OpenAI API | 各 provider 独立 client | 统一 `AsyncOpenAI(base_url=...)` |
| 持久化 | 部分（recall / archival） | 全部（blocks / messages / archives 都入库） | recall + archival 入库；core memory 仅内存 |

## 附：文件对照表

| Letta | mini |
|---|---|
| [letta/schemas/block.py](/data/letta/letta/schemas/block.py) + [letta/orm/block.py](/data/letta/letta/orm/block.py) + [letta/services/block_manager.py](/data/letta/letta/services/block_manager.py) | [memgpt/memory/core_memory.py](../memgpt/memory/core_memory.py) 一个文件 |
| [letta/services/passage_manager.py](/data/letta/letta/services/passage_manager.py) + [letta/services/archive_manager.py](/data/letta/letta/services/archive_manager.py) | [memgpt/memory/archival.py](../memgpt/memory/archival.py) |
| [letta/services/message_manager.py](/data/letta/letta/services/message_manager.py) | [memgpt/memory/recall.py](../memgpt/memory/recall.py) |
| [letta/services/summarizer/*](/data/letta/letta/services/summarizer/) (~2000 行) | [memgpt/compaction.py](../memgpt/compaction.py) (~260 行) |
| [letta/functions/function_sets/base.py](/data/letta/letta/functions/function_sets/base.py) | [memgpt/tools/base_tools.py](../memgpt/tools/base_tools.py) |
| [letta/agents/letta_agent_v3.py](/data/letta/letta/agents/letta_agent_v3.py) (~2500 行) | [memgpt/agent.py](../memgpt/agent.py) (~180 行) |

```
Letta 总代码   ~数万行
mini 总代码    ~900 行
```

这 900 行足够跑出论文 Figure 1 的所有核心行为：multi-block 工作记忆、持久化归档、召回搜索、压缩、三级协作。不够的是：多 provider、审计、团队协作、sandbox——这些生产特性正是 Letta 的护城河，也是 mini 不碰的部分。
