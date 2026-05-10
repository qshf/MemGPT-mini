"""FIFO queue compaction. Mirrors Letta's services/summarizer/* in miniature.

Default mode is ``sliding_window`` (Letta's default), with ``all`` as fallback.
Layout after compaction: ``[system, summary_as_user, *tail]``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import tiktoken
from openai import AsyncOpenAI

from memgpt.config import Config, chat_extras


# ---- Prompts ---------------------------------------------------------------
# Trimmed but behavior-equivalent to Letta's SLIDING_PROMPT / ALL_PROMPT
# (see /data/letta/letta/prompts/summarizer_prompt.py).

SLIDING_PROMPT = """The following messages are being evicted from the BEGINNING \
of your context window. Write a detailed summary that appears BEFORE the \
remaining recent messages, providing background for what comes after. \
Include these sections:

1. High-level goals — the user's explicit requests and intent.
2. What happened — conversations and actions in order.
3. Important details — identifiers, numbers, file paths preserved verbatim.
4. Errors and fixes — verbatim user feedback where useful.
5. Lookup hints — topics and keywords for finding evicted content in recall memory.

Write in first person. Under 300 words. Output only the summary."""

ALL_PROMPT = """Your task is to create a detailed summary of the conversation so \
far. Include high-level goals, what happened, important details (preserve \
identifiers verbatim), errors and fixes, the current state, optional next step, \
and lookup hints. Under 500 words. Output only the summary."""

MESSAGE_SUMMARY_REQUEST_ACK = (
    "Understood, I will respond with a summary of the message (and only the "
    "summary, nothing else) once I receive the conversation history. I'm ready."
)

SUMMARY_TRUNCATION_SUFFIX = " ... [summary truncated to fit]"


# ---- Token counting --------------------------------------------------------


def count_tokens(messages: list[dict[str, Any]], model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for m in messages:
        total += 4  # per-message framing
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, str):
                total += len(enc.encode(v))
            else:
                total += len(enc.encode(str(v)))
    return total + 2


def count_tokens_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str = "gpt-4o-mini",
) -> int:
    """Like ``count_tokens`` but also accounts for the tool schemas attached to
    every chat.completions call. Letta does the same via ``count_tool_tokens``
    in its summarizer to prevent the post-compact estimate from undercounting."""
    base = count_tokens(messages, model)
    if not tools:
        return base
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tool_tokens = 0
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name") or ""
        desc = fn.get("description") or ""
        params = str(fn.get("parameters") or {})
        tool_tokens += len(enc.encode(name)) + len(enc.encode(desc)) + len(enc.encode(params)) + 3
    return base + tool_tokens


# ---- Cutoff selection ------------------------------------------------------


def _is_valid_cutoff(m: dict[str, Any]) -> bool:
    """Letta rule: tail must begin with an ``assistant`` message so any preceding
    ``tool_calls``/``tool`` pair remains complete on the evicted side. (Letta
    also permits ``approval`` with tool_calls; mini has no approval role.)"""
    return m.get("role") == "assistant"


def _sliding_window_cutoff(
    messages: list[dict[str, Any]],
    cfg: Config,
    tools: list[dict[str, Any]] | None,
) -> int | None:
    """Find the index where tail should start. Mirrors Letta's inner loop in
    ``summarize_via_sliding_window`` (summarizer_sliding_window.py:163-191).

    Starts at ``sliding_window_percentage`` of the message list and walks the
    eviction ratio up in +10% steps, each time locating the *latest* assistant
    message within the candidate eviction range. Stops when the retained
    ``[system, *tail]`` fits under ``goal_tokens``.
    """
    n = len(messages)
    goal_tokens = int((1 - cfg.sliding_window_percentage) * cfg.context_window)
    eviction_pct = cfg.sliding_window_percentage
    approx = cfg.context_window
    chosen: int | None = None
    while approx >= goal_tokens and eviction_pct < 1.0:
        eviction_pct += 0.10
        cutoff_idx = min(round(eviction_pct * n), n - 1)
        chosen = next(
            (i for i in reversed(range(1, cutoff_idx + 1)) if _is_valid_cutoff(messages[i])),
            None,
        )
        if chosen is None:
            continue
        approx = count_tokens_with_tools([messages[0], *messages[chosen:]], tools, cfg.model)
    return chosen


# ---- Rendering (reused for transcript + legacy tests) ----------------------


def _render(messages_slice: list[dict[str, Any]] | dict[str, Any]) -> str:
    """Render one or many messages using sibling tool_calls to resolve tool_name.
    Backward-compat: accepts a single dict for the old per-message rendering."""
    if isinstance(messages_slice, dict):
        return _render_one(messages_slice, name_by_id={})
    name_by_id: dict[str, str] = {}
    for m in messages_slice:
        for tc in m.get("tool_calls") or []:
            name_by_id[tc["id"]] = tc["function"]["name"]
    return "\n\n".join(_render_one(m, name_by_id) for m in messages_slice)


def _render_one(m: dict[str, Any], name_by_id: dict[str, str]) -> str:
    role = m.get("role", "?")
    content = m.get("content") or ""
    if m.get("tool_calls"):
        calls = ", ".join(c["function"]["name"] for c in m["tool_calls"])
        content = (content + f"\n[tool_calls: {calls}]").strip()
    if m.get("tool_call_id"):
        tool_name = name_by_id.get(m["tool_call_id"], "?")
        content = f"[tool_result {tool_name}] {content}"
    return f"[{role}] {content}"


# ---- Summary LLM call ------------------------------------------------------


async def _simple_summary(
    evicted: list[dict[str, Any]],
    prompt: str,
    cfg: Config,
    client: AsyncOpenAI,
) -> str:
    transcript = _render(evicted)
    input_messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
    if cfg.include_summary_ack:
        input_messages.append({"role": "assistant", "content": MESSAGE_SUMMARY_REQUEST_ACK})
    input_messages.append(
        {
            "role": "user",
            "content": f"<start_transcript>\n{transcript}\n<end_transcript>\nGenerate the summary.",
        }
    )
    resp = await client.chat.completions.create(
        model=cfg.model,
        messages=input_messages,
        temperature=0.2,
        **chat_extras(cfg),
    )
    return (resp.choices[0].message.content or "").strip()


# ---- Public API ------------------------------------------------------------


async def compact(
    messages: list[dict[str, Any]],
    cfg: Config,
    client: AsyncOpenAI,
    trigger_threshold: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Compact an in-context message list. Returns ``(new_messages, summary_text)``.

    Layout mirrors Letta: ``[system, summary_as_user, *tail]``.

    Mode selection:
      - ``sliding_window`` (default): pick an assistant-message cutoff, evict
        everything between system and cutoff, keep everything from cutoff on.
      - ``all``: evict the entire history except system; tail is empty.

    ``trigger_threshold`` enables Letta's post-compact guarantee: if after a
    sliding_window pass the result is still at/above the threshold, retry in
    ``all`` mode.
    """
    if len(messages) <= 2:
        return messages, ""

    mode = cfg.summarization_mode
    evict: list[dict[str, Any]]
    tail: list[dict[str, Any]]

    if mode == "sliding_window":
        idx = _sliding_window_cutoff(messages, cfg, tools)
        if idx is None or idx >= len(messages):
            # No valid assistant cutoff exists — fall back to `all`.
            mode = "all"
            evict, tail = messages[1:], []
            prompt = ALL_PROMPT
        else:
            evict, tail = messages[1:idx], messages[idx:]
            prompt = SLIDING_PROMPT
    elif mode == "all":
        evict, tail = messages[1:], []
        prompt = ALL_PROMPT
    else:
        raise ValueError(f"Unknown summarization_mode: {mode!r}")

    if not evict:
        return messages, ""

    summary_text = await _simple_summary(evict, prompt, cfg, client)
    if cfg.summarizer_clip_chars and len(summary_text) > cfg.summarizer_clip_chars:
        summary_text = summary_text[: cfg.summarizer_clip_chars] + SUMMARY_TRUNCATION_SUFFIX

    n_evicted = len(evict)
    summary_msg = {
        "role": "user",
        "content": (
            f"Note: {n_evicted} prior message(s) have been hidden from view due to "
            f"conversation memory constraints. The following is a summary of the "
            f"previous messages:\n{summary_text}"
        ),
    }
    new_messages = [messages[0], summary_msg, *tail]

    # Post-compact guarantee (Letta compact.py:360-412): if still over the
    # trigger threshold after sliding_window, re-run in `all` mode against the
    # already-shrunken messages to make a second, more aggressive pass.
    if trigger_threshold and mode == "sliding_window":
        after = count_tokens_with_tools(new_messages, tools, cfg.model)
        if after >= trigger_threshold:
            retry_cfg = replace(cfg, summarization_mode="all")
            return await compact(
                [messages[0], *tail] if tail else [messages[0], summary_msg],
                retry_cfg,
                client,
                trigger_threshold=None,
                tools=tools,
            )

    return new_messages, summary_text
