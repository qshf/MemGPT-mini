"""End-to-end narrative test with full tracing.

Runs a scripted multi-turn conversation that exercises all 3 memory tiers,
triggers compaction, and forces post-compaction recall. Every step prints:
  - system prompt (rendered core_memory)
  - in_context (FIFO) before the LLM call
  - LLM request metadata + response
  - tool calls and their results
  - state snapshot after the turn

Output is tee'd to tests/story_run.log.

Run:  uv run python tests/test_story.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memgpt import agent as agent_module
from memgpt.agent import MemGPTAgent
from memgpt.config import load_config
from memgpt.db import Database
from memgpt.tools import base_tools


LOG_PATH = Path(__file__).parent / "story_run.log"
log_fp = open(LOG_PATH, "w")


def out(s: str = "") -> None:
    print(s)
    log_fp.write(s + "\n")
    log_fp.flush()


def hr(char: str = "=", width: int = 100) -> None:
    out(char * width)


def banner(title: str) -> None:
    hr()
    out(title)
    hr()


def indent(s: str, prefix: str = "  ") -> str:
    return textwrap.indent(s, prefix)


def short(s: str | None, n: int = 180) -> str:
    if s is None:
        return "<none>"
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


# ---------- instrumentation ----------
llm_call_counter = 0


def install_tracers(agent: MemGPTAgent) -> None:
    """Monkey-patch the client and the dispatch function to log every call."""
    orig_create = agent.client.chat.completions.create
    orig_dispatch = base_tools.dispatch

    async def traced_create(**kwargs):
        global llm_call_counter
        llm_call_counter += 1
        idx = llm_call_counter
        msgs = kwargs.get("messages", [])
        out(f"\n  >>> LLM CALL #{idx}  model={kwargs.get('model')} messages={len(msgs)} tools={len(kwargs.get('tools', []))}")
        # System prompt snapshot (first 300 chars)
        if msgs and msgs[0].get("role") == "system":
            out(f"      system[0]: {short(msgs[0]['content'], 300)}")
        # Last 3 messages preview
        for i, m in enumerate(msgs[-3:]):
            idx_real = len(msgs) - 3 + i if len(msgs) > 3 else i
            role = m.get("role", "?")
            content = short(m.get("content"))
            tcs = m.get("tool_calls")
            tc_hint = f" tool_calls={[t['function']['name'] for t in tcs]}" if tcs else ""
            tc_id = f" tool_call_id={m['tool_call_id']}" if m.get("tool_call_id") else ""
            tool_name = f" tool_name={m['tool_name']}" if m.get("tool_name") else ""
            out(f"      [{idx_real}] {role}{tc_hint}{tc_id}{tool_name}: {content}")
        resp = await orig_create(**kwargs)
        choice = resp.choices[0].message
        usage = resp.usage
        out(f"  <<< LLM RESP #{idx}  tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
        if choice.content:
            out(f"      content: {short(choice.content, 250)}")
        if choice.tool_calls:
            for tc in choice.tool_calls:
                out(f"      tool_call: {tc.function.name}({short(tc.function.arguments, 200)})")
        return resp

    async def traced_dispatch(a, name, args):
        out(f"  >> TOOL {name}  args={short(json.dumps(args, ensure_ascii=False), 250)}")
        result = await orig_dispatch(a, name, args)
        out(f"  << TOOL {name}  -> {short(result, 250)}")
        return result

    agent.client.chat.completions.create = traced_create
    # Must patch at the import site used by agent.py
    agent_module.dispatch = traced_dispatch


def dump_state(agent: MemGPTAgent) -> None:
    out("  [core memory]")
    for label, b in agent.core_memory.blocks.items():
        out(f"    {label} ({len(b.value)}/{b.limit}): {short(b.value, 220)}")
    out(f"  [in_context]  {len(agent.in_context)} messages")
    for i, m in enumerate(agent.in_context):
        role = m.get("role", "?")
        content = short(m.get("content"), 120)
        tcs = m.get("tool_calls")
        hint = f" tool_calls={[t['function']['name'] for t in tcs]}" if tcs else ""
        tc_id = f" tool_call_id={m['tool_call_id']}" if m.get("tool_call_id") else ""
        tool_name = f" tool_name={m['tool_name']}" if m.get("tool_name") else ""
        out(f"    [{i}] {role}{hint}{tc_id}{tool_name}: {content}")


# ---------- the story ----------
SCRIPT = [
    (
        "identity",
        "你好，我是张伟，Acme 公司的资深工程师，负责认证服务。请记住我的名字和职位。",
    ),
    (
        "archival-fact-1",
        "请把这些技术细节存到长期记忆里（归档后简短确认即可，不要在回复里复述具体数值）："
        "我们的认证服务峰值 QPS 是 217 万；JWT 使用 RS256 签名；access token 有效期 24 小时；"
        "refresh token 有效期 30 天；密钥轮换周期 90 天。",
    ),
    (
        "archival-fact-2",
        "再存一条：我的值班时间是周一、周三、周五；备班是陈晓；升级群是飞书的 #认证值班。"
        "归档后简短确认，不要复述。",
    ),
    (
        "smalltalk",
        "对了随便聊一句，我昨晚看完了《三体》动画第 8 集，感觉还可以。不用做任何记忆操作，直接闲聊一句就好。",
    ),
    (
        "filler-1",
        "换个话题：请系统性地讲一下高流量 API 用 JWT 和 opaque session token 的取舍。",
    ),
    (
        "filler-2",
        "接着问：从运维角度，无状态认证和有状态认证各有什么优劣？重点讲失效模式。",
    ),
    (
        "filler-3",
        "最后一个：refresh token rotation 是怎么防止 token 被盗用的？假设攻击者已经拿到了网络层访问权。",
    ),
    (
        "recall-core",
        "问一下——我叫什么名字？",
    ),
    (
        "recall-archival-specific-1",
        "我们认证服务峰值 QPS 具体是多少？access token 的有效期呢？",
    ),
    (
        "recall-archival-specific-2",
        "我不在的时候谁是备班？升级群叫什么？",
    ),
    (
        "recall-archival-specific-3",
        "我们 JWT 签名算法是什么？密钥多久轮换一次？",
    ),
    (
        "recall-conversation-only",
        "之前我有跟你提过我在看什么动画吗？具体看到第几集了？这些细节我没让你存档，所以需要你去翻一下我们之前的对话记录。",
    ),
]


async def main() -> None:
    cfg = load_config()
    db = Database(cfg)
    await db.init_schema()

    banner("SETUP")
    out(f"timestamp:        {datetime.now().isoformat(timespec='seconds')}")
    out(f"chat model:       {cfg.model} @ {cfg.chat_base_url}")
    out(f"embed model:      {cfg.embed_model} @ {cfg.embed_base_url} (dim={cfg.embed_dim})")
    out(f"context_window:   {cfg.context_window}  trigger={cfg.summarization_trigger}  tail_keep={cfg.message_buffer_min}")
    out(f"postgres:         {cfg.pg_uri}")
    out(f"log file:         {LOG_PATH}")

    agent = MemGPTAgent(cfg=cfg, db=db, agent_id=f"story-{datetime.now():%H%M%S}")
    install_tracers(agent)

    banner("INITIAL STATE (before any user turn)")
    out("[system prompt skeleton — rendered each turn]")
    out(indent(agent._build_system_message()["content"], "  "))
    dump_state(agent)

    for turn_idx, (label, user_msg) in enumerate(SCRIPT):
        banner(f"TURN {turn_idx}  [{label}]")
        out(f"USER: {user_msg}")
        reply = await agent.step(user_msg)
        out(f"\nAGENT: {reply}")
        if agent.last_compaction_summary:
            out(f"\n*** COMPACTION FIRED THIS TURN ***")
            out(f"summary: {agent.last_compaction_summary}")
        out("")
        dump_state(agent)

    banner("DONE")
    out(f"total LLM calls: {llm_call_counter}")
    out(f"final in_context: {len(agent.in_context)} messages")
    from memgpt.db import MessageRow
    from sqlalchemy import func, select
    async with db.session() as s:
        n_msgs = (await s.execute(select(func.count(MessageRow.id)).where(MessageRow.agent_id == agent.agent_id))).scalar_one()
        from memgpt.db import Passage
        n_pass = (await s.execute(select(func.count(Passage.id)))).scalar_one()
    out(f"recall (DB messages for this agent): {n_msgs}")
    out(f"archival (DB passages total): {n_pass}")
    log_fp.close()


if __name__ == "__main__":
    asyncio.run(main())
