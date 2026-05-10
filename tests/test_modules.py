"""Module-by-module smoke tests for memgpt-mini.

Run:  uv run python tests/test_modules.py
Run one: uv run python tests/test_modules.py core   (or recall/archival/tools/compact/agent)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Callable, Awaitable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memgpt.agent import MemGPTAgent
from memgpt.compaction import compact, count_tokens
from memgpt.config import load_config
from memgpt.db import Database
from memgpt.memory.archival import ArchivalMemory
from memgpt.memory.core_memory import Block, CoreMemory
from memgpt.memory.recall import RecallMemory


def ok(msg: str) -> None:
    print(f"  \033[32mOK\033[0m {msg}")


def header(name: str) -> None:
    print(f"\n=== {name} ===")


# ------------------------------------------------------------------ core memory
async def test_core_memory() -> None:
    header("core_memory: in-RAM working context")
    cm = CoreMemory()
    cm.add(Block(label="human", limit=100))
    cm.add(Block(label="persona", value="I am mini.", limit=100))

    cm.append("human", "Alice is a Go dev.")
    cm.append("human", "Prefers concise answers.")
    assert "Alice" in cm.get("human").value
    ok(f"append: {cm.get('human').value!r}")

    cm.replace("human", "Go dev", "Go developer")
    assert "Go developer" in cm.get("human").value
    ok(f"replace: {cm.get('human').value!r}")

    try:
        cm.append("human", "x" * 200)
    except ValueError as e:
        ok(f"limit enforced: {e}")

    rendered = cm.compile()
    assert "<core_memory>" in rendered and "persona" in rendered and "human" in rendered
    ok("compile produces <core_memory> XML with both blocks")


# ------------------------------------------------------------------ recall
async def test_recall(db: Database) -> None:
    header("recall: Postgres full message log")
    recall = RecallMemory(db, agent_id="test-recall")
    for m in [
        {"role": "user", "content": "hello from Alice"},
        {"role": "assistant", "content": "Hi Alice!"},
        {"role": "user", "content": "I like chess"},
        {"role": "assistant", "content": "Nice, what opening?"},
    ]:
        await recall.append(m)
    ok(f"appended 4 messages (total={await recall.size()})")

    hits = await recall.search(query="chess", limit=5)
    assert any("chess" in h.content for h in hits)
    ok(f"ILIKE search 'chess' -> {len(hits)} hit(s): {hits[0].content!r}")

    hits = await recall.search(roles=["user"], limit=10)
    assert all(h.role == "user" for h in hits)
    ok(f"role filter 'user' -> {len(hits)} hit(s)")


# ------------------------------------------------------------------ archival
async def test_archival(db: Database) -> None:
    header("archival: pgvector semantic memory + archive_id isolation")
    cfg = load_config()
    arch_a = ArchivalMemory(cfg, db, archive_id="test-archival-a")
    arch_b = ArchivalMemory(cfg, db, archive_id="test-archival-b")
    for t in [
        "Alice is a senior Go developer who loves distributed systems.",
        "Alice's favorite board game is chess; she plays the Sicilian Defense.",
        "The office coffee machine is broken until Friday.",
    ]:
        await arch_a.insert(t)
    await arch_b.insert("Bob's secret passphrase is frobnicate.")
    ok("inserted 3 passages in archive A + 1 in archive B")

    results = await arch_a.search("what programming language does the user use?", top_k=3)
    assert any("Go" in r.text for r in results)
    assert not any("Bob" in r.text for r in results)
    ok(f"archive A search -> top={results[0].text!r} score={results[0].score:.3f} (no cross-archive leak)")

    results = await arch_a.search("hobbies")
    assert any("chess" in r.text for r in results)
    ok(f"query 'hobbies' -> top={results[0].text!r} score={results[0].score:.3f}")

    leak = await arch_a.search("passphrase", top_k=5)
    assert all("Bob" not in r.text for r in leak), "archive A leaked archive B's data"
    ok(f"archive isolation verified: searching for B's data from A returns {len(leak)} unrelated hits")


# ------------------------------------------------------------------ tool dispatch
async def test_tools(db: Database) -> None:
    header("tools: dispatch all 6 tools against a live agent")
    from memgpt.tools.base_tools import dispatch
    cfg = load_config()
    agent = MemGPTAgent(cfg=cfg, db=db, agent_id="test-tools")

    r = await dispatch(agent, "core_memory_append", {"label": "human", "content": "Name: Bob"})
    assert "Name: Bob" in r
    ok(f"core_memory_append -> returns full block value ({r!r})")
    r = await dispatch(agent, "core_memory_replace", {"label": "human", "old_content": "Bob", "new_content": "Robert"})
    assert "Robert" in r and "Bob" not in r
    ok(f"core_memory_replace -> returns full block value ({r!r})")
    r = await dispatch(agent, "archival_memory_insert", {"content": "Robert likes go fishing on weekends."})
    ok(f"archival_memory_insert -> {r}")
    r = await dispatch(agent, "archival_memory_search", {"query": "weekend hobbies", "top_k": 2})
    ok(f"archival_memory_search -> first line: {r.splitlines()[0]}")
    await agent.recall.append({"role": "user", "content": "The secret passphrase is hippogriff"})
    r = await dispatch(agent, "conversation_search", {"query": "passphrase"})
    ok(f"conversation_search -> {r.splitlines()[0]}")
    r = await dispatch(agent, "send_message", {"message": "hello"})
    assert agent.last_send_message == "hello"
    ok("send_message sets agent.last_send_message")


async def test_tool_message_metadata(db: Database) -> None:
    header("agent: in_context is LLM-clean; tool_name lives only in the DB")
    cfg = load_config()
    agent = MemGPTAgent(cfg=cfg, db=db, agent_id="test-tool-name")

    tool_msg = {
        "role": "tool",
        "tool_call_id": "call_demo",
        "content": "Appended. Block 'human' now 10 chars.",
    }
    agent.in_context.append(tool_msg)
    await agent.recall.append(tool_msg, tool_name="core_memory_append")

    # in_context message carries only standard OpenAI fields
    assert "tool_name" not in agent.in_context[-1]
    ok("in_context message has no non-standard fields")

    # messages_for_llm returns them unchanged
    llm_messages = agent._messages_for_llm()
    assert llm_messages[-1]["tool_call_id"] == "call_demo"
    assert "tool_name" not in llm_messages[-1]
    ok("LLM payload is OpenAI-schema clean")

    # DB row has tool_name for debugging
    hits = await agent.recall.search(query="Appended", limit=5)
    assert any(h.tool_name == "core_memory_append" for h in hits)
    ok("DB row preserves tool_name for conversation_search readability")


# ------------------------------------------------------------------ compaction
async def test_compaction() -> None:
    header("compaction: sliding_window keeps a clean tail starting at an assistant message")
    cfg = load_config()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=cfg.chat_api_key, base_url=cfg.chat_base_url)

    # Interleave plain turns with a tool_calls / tool pair so we can verify the
    # cutoff does NOT cut the pair.
    messages: list[dict] = [{"role": "system", "content": "system prompt."}]
    for i in range(8):
        messages.append({"role": "user", "content": f"Message {i}: random filler about topic #{i}."})
        messages.append({"role": "assistant", "content": f"Reply {i}: acknowledging topic #{i}."})
    # Inject a tool_calls / tool_result pair near the end
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_demo", "type": "function", "function": {"name": "archival_memory_search", "arguments": "{}"}}
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "call_demo", "content": "found: nothing"})
    messages.append({"role": "assistant", "content": "Final reply after tool."})

    tokens_before = count_tokens(messages, cfg.model)
    ok(f"built {len(messages)} messages, ~{tokens_before} tokens")

    # Force the cutoff loop to evict a sizeable prefix by pretending the window is small.
    from dataclasses import replace
    small_cfg = replace(cfg, context_window=max(tokens_before * 2, 1500), sliding_window_percentage=0.3)
    new_messages, summary = await compact(new_messages := messages, small_cfg, client)
    ok(f"compact -> {len(new_messages)} messages (layout: [system, summary, ...tail])")

    assert new_messages[0]["role"] == "system"
    assert "hidden from view" in new_messages[1]["content"]
    ok(f"summary role={new_messages[1]['role']}, first summary line: {summary.splitlines()[0][:120]!r}")

    # Tail integrity: if the tool message exists in tail, its preceding assistant with
    # matching tool_calls must also be in tail (i.e. the pair wasn't split).
    tail = new_messages[2:]
    assert tail, "tail should be non-empty for sliding_window"
    assert tail[0]["role"] == "assistant", f"tail must start with assistant, got {tail[0]['role']}"
    ok(f"tail starts with assistant (index 2): {str(tail[0].get('content'))[:80]!r}")

    tool_ids_in_tail = {m["tool_call_id"] for m in tail if m.get("tool_call_id")}
    if tool_ids_in_tail:
        issued = set()
        for m in tail:
            for tc in m.get("tool_calls") or []:
                issued.add(tc["id"])
        assert tool_ids_in_tail.issubset(issued), (
            f"orphan tool results in tail: {tool_ids_in_tail - issued}"
        )
        ok(f"tool_call pair integrity preserved: {tool_ids_in_tail}")
    else:
        ok("tool pair was fully evicted to the summary side (also valid)")


# ------------------------------------------------------------------ sliding_window cutoff unit test (no LLM)
async def test_sliding_window_cutoff() -> None:
    header("compaction: _sliding_window_cutoff picks the latest assistant index")
    from memgpt.compaction import _sliding_window_cutoff
    from dataclasses import replace

    cfg = load_config()

    # 20 alternating messages + 1 tool pair
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(8):
        msgs.append({"role": "user", "content": f"u{i}" * 40})  # make each user sizeable
        msgs.append({"role": "assistant", "content": f"a{i}" * 40})
    msgs.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        }
    )
    msgs.append({"role": "tool", "tool_call_id": "c1", "content": "ok"})
    msgs.append({"role": "assistant", "content": "done"})

    # Tight window forces cutoff to move forward
    tight_cfg = replace(cfg, context_window=400, sliding_window_percentage=0.3)
    idx = _sliding_window_cutoff(msgs, tight_cfg, tools=None)
    assert idx is not None, "expected a cutoff to be found"
    assert msgs[idx]["role"] == "assistant", f"cutoff must land on assistant, got role={msgs[idx]['role']}"
    ok(f"cutoff idx={idx} -> role=assistant  content={str(msgs[idx].get('content'))[:60]!r}")

    tail = msgs[idx:]
    assert tail[0]["role"] == "assistant" and tail[0]["role"] != "tool"
    ok(f"tail[0] is assistant (not tool, not user), len(tail)={len(tail)}")

    # Very generous window → no compaction needed; cutoff may still be set but
    # approx tokens will already be under goal on the first loop iteration.
    loose_cfg = replace(cfg, context_window=1_000_000, sliding_window_percentage=0.3)
    idx2 = _sliding_window_cutoff(msgs, loose_cfg, tools=None)
    # Under Letta's algorithm, the loop still runs once and lands on a cutoff.
    if idx2 is not None:
        assert msgs[idx2]["role"] == "assistant"
        ok(f"loose window: still returns an assistant cutoff at idx={idx2}")


# ------------------------------------------------------------------ post-compact fallback
async def test_post_compact_fallback() -> None:
    header("compaction: post-compact threshold check falls back to 'all' mode")
    from memgpt import compaction as comp
    from dataclasses import replace

    cfg = load_config()

    # Build a message list that will clearly exceed a fake threshold.
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"u{i}" * 100})
        msgs.append({"role": "assistant", "content": f"a{i}" * 100})

    call_order: list[str] = []

    async def fake_summary(evicted, prompt, c, client):
        # Record which prompt was used so we can assert the mode sequence.
        tag = "sliding" if "BEGINNING" in prompt else "all"
        call_order.append(tag)
        # Return a pathologically long summary on the first (sliding) pass so the
        # post-compact token check still exceeds the threshold → forces 'all' retry.
        if tag == "sliding":
            return "x" * 40000
        return "concise final summary."

    orig = comp._simple_summary
    comp._simple_summary = fake_summary
    try:
        tight_cfg = replace(cfg, context_window=2000, sliding_window_percentage=0.3)
        # Use a threshold that's below the (inflated) sliding-mode output but above the 'all' output.
        new_messages, summary = await comp.compact(
            msgs, tight_cfg, client=None, trigger_threshold=1500, tools=None,
        )
    finally:
        comp._simple_summary = orig

    assert call_order[0] == "sliding", f"expected sliding first, got {call_order}"
    assert "all" in call_order, f"expected fallback to 'all', got {call_order}"
    ok(f"sequence: {call_order} (sliding_window -> all fallback)")
    assert "concise final summary" in summary
    ok(f"final summary is the 'all'-mode one: {summary!r}")


# ------------------------------------------------------------------ end-to-end compaction via the agent
async def test_agent_compaction(db: Database) -> None:
    header("agent: trigger compaction by exceeding context_window (.env should be small, e.g. 4000)")
    cfg = load_config()
    print(f"  context_window={cfg.context_window}  trigger={cfg.summarization_trigger}")
    agent = MemGPTAgent(cfg=cfg, db=db, agent_id="test-compact")

    filler = "Here is a long factoid the user wants remembered: " + ("lorem ipsum " * 80)
    for i in range(8):
        reply = await agent.step(f"Turn {i}: {filler}")
        n = len(agent.in_context)
        print(f"  turn {i}: in_context={n} summary_this_turn={bool(agent.last_compaction_summary)}")
        if agent.last_compaction_summary:
            ok(f"compaction fired on turn {i}: {agent.last_compaction_summary[:140]!r}...")
            break
    else:
        print("  (no compaction fired — try a smaller MEMGPT_CONTEXT_WINDOW)")


# ------------------------------------------------------------------ runner
TESTS: dict[str, Callable[..., Awaitable[None]]] = {
    "core": test_core_memory,
    "recall": test_recall,
    "archival": test_archival,
    "tools": test_tools,
    "toolmeta": test_tool_message_metadata,
    "compact": test_compaction,
    "cutoff": test_sliding_window_cutoff,
    "fallback": test_post_compact_fallback,
    "agent": test_agent_compaction,
}


async def main() -> None:
    cfg = load_config()
    db = Database(cfg)
    await db.init_schema()

    wanted = sys.argv[1:] or list(TESTS)
    for name in wanted:
        fn = TESTS[name]
        # Dispatch by arg signature
        if name in ("recall", "archival", "tools", "toolmeta", "agent"):
            await fn(db)
        else:
            await fn()
    print("\n\033[32mdone.\033[0m")


if __name__ == "__main__":
    asyncio.run(main())
