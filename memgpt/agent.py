"""MemGPT agent step loop. Mirrors Letta's LettaAgentV3 in miniature."""
from __future__ import annotations

import json
import uuid
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from memgpt.compaction import compact, count_tokens
from memgpt.config import Config, chat_extras
from memgpt.db import Database
from memgpt.memory.archival import ArchivalMemory
from memgpt.memory.core_memory import Block, CoreMemory
from memgpt.memory.recall import RecallMemory
from memgpt.tools.base_tools import TOOL_SCHEMAS, dispatch


DEFAULT_SYSTEM_PROMPT = """You are MemGPT-mini, a stateful assistant with hierarchical memory.

You have three memory tiers, all of which are YOURS to manage:
  1. Core memory (always in context): small, structured blocks rendered in <core_memory>.
     Edit with core_memory_append / core_memory_replace when you learn durable facts about the user.
  2. Archival memory (vector-searched): unbounded long-term store.
     Write with archival_memory_insert for technical specs, project details, reference data that
     won't fit in core memory. Read with archival_memory_search.
  3. Recall memory (full message log): conversation_search surfaces older messages.

Rules of thumb:
- When the [PRIOR CONVERSATION SUMMARY] hints that details were stored in archival, DO NOT guess
  from the summary — call archival_memory_search with a relevant query and use its result.
- If the user asks about something that was mentioned in chat but NOT stored in core or archival
  (e.g. casual side remarks, things you didn't deem important enough to save at the time), use
  conversation_search to scan the full message log — that's recall memory's whole purpose.
- When you insert something into archival, reply briefly ("Got it, stored.") rather than echoing
  the full content back; the data lives in archival now, not the conversation.
- Always end each turn by replying to the user via plain assistant content or send_message.
"""


class MemGPTAgent:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        agent_id: str | None = None,
        archive_id: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        core_memory: CoreMemory | None = None,
        max_tool_steps: int = 6,
    ):
        self.cfg = cfg
        self.db = db
        self.agent_id = agent_id or str(uuid.uuid4())
        # Default to a private archive per agent. Pass an explicit ``archive_id`` to share
        # one archive across multiple agents (team knowledge base pattern in Letta).
        self.archive_id = archive_id or self.agent_id
        self.system_prompt = system_prompt
        self.core_memory = core_memory or _default_core_memory()
        self.client = AsyncOpenAI(api_key=cfg.chat_api_key, base_url=cfg.chat_base_url)
        self.archival = ArchivalMemory(cfg, db, archive_id=self.archive_id)
        self.recall = RecallMemory(db, self.agent_id)
        self.in_context: list[dict[str, Any]] = []
        self.max_tool_steps = max_tool_steps
        self.last_send_message: str | None = None
        self.last_compaction_summary: str | None = None

    def _build_system_message(self) -> dict[str, Any]:
        return {
            "role": "system",
            "content": f"{self.system_prompt}\n\n{self.core_memory.compile()}",
        }

    def _messages_for_llm(self) -> list[dict[str, Any]]:
        return [self._build_system_message(), *self.in_context]

    async def step(self, user_message: str) -> str:
        """Run one user-turn. Returns the send_message text."""
        self.last_send_message = None
        self.last_compaction_summary = None

        user_msg = {"role": "user", "content": user_message}
        self.in_context.append(user_msg)
        await self.recall.append(user_msg)

        for _ in range(self.max_tool_steps):
            messages = self._messages_for_llm()
            try:
                resp = await self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    **chat_extras(self.cfg),
                )
            except BadRequestError as e:
                if "context" in str(e).lower() or "token" in str(e).lower():
                    await self._run_compaction(trigger="context_window_exceeded")
                    continue
                raise

            choice = resp.choices[0].message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": choice.content}
            # DeepSeek thinking mode requires echoing reasoning_content back on the next call.
            reasoning = getattr(choice, "reasoning_content", None) or (choice.model_extra or {}).get("reasoning_content")
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            if choice.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.tool_calls
                ]
            self.in_context.append(assistant_msg)
            await self.recall.append(assistant_msg)

            if not choice.tool_calls:
                break

            for tc in choice.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = await dispatch(self, name, args)
                except Exception as e:
                    result = f"ERROR: {e}"
                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
                self.in_context.append(tool_msg)
                await self.recall.append(tool_msg, tool_name=name)

            if self.last_send_message is not None:
                break

        await self._maybe_compact(resp_usage=resp.usage)
        # Fallback: if the model replied via plain content instead of send_message, use that.
        if self.last_send_message is None:
            last = next((m for m in reversed(self.in_context) if m.get("role") == "assistant" and m.get("content")), None)
            if last:
                self.last_send_message = last["content"]
        return self.last_send_message or "(no reply)"

    async def _maybe_compact(self, resp_usage: Any) -> None:
        """Post-step compaction check. Mirrors Letta V3 段 14."""
        threshold = int(self.cfg.context_window * self.cfg.summarization_trigger)
        used = getattr(resp_usage, "total_tokens", None) or count_tokens(self._messages_for_llm(), self.cfg.model)
        if used > threshold:
            await self._run_compaction(trigger="post_step_context_check", used=used, threshold=threshold)

    async def _run_compaction(self, trigger: str, used: int | None = None, threshold: int | None = None) -> None:
        before = len(self.in_context)
        messages = [self._build_system_message(), *self.in_context]
        new_messages, summary = await compact(
            messages,
            self.cfg,
            self.client,
            trigger_threshold=threshold,
            tools=TOOL_SCHEMAS,
        )
        self.in_context = new_messages[1:]  # drop system; regenerated each turn
        self.last_compaction_summary = summary
        print(
            f"[compaction:{trigger}] mode={self.cfg.summarization_mode} used={used} "
            f"threshold={threshold} messages {before} -> {len(self.in_context)}"
        )


def _default_core_memory() -> CoreMemory:
    cm = CoreMemory()
    cm.add(Block(label="persona", description="Who the assistant is.", value="I am MemGPT-mini, a helpful assistant with persistent memory."))
    cm.add(Block(label="human", description="What I know about the user.", value=""))
    return cm
