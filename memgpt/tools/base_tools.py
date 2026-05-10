"""Tool schemas + dispatch.

Signatures and docstrings mirror Letta's base tool set
(letta/functions/function_sets/base.py). The tool schemas below are what we
actually advertise to the LLM via the chat.completions `tools=` parameter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memgpt.agent import MemGPTAgent


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Sends a message to the human user. All unicode (including emojis) is supported.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Message contents."}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_append",
            "description": "Append to the contents of core memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Section of the memory to be edited."},
                    "content": {"type": "string", "description": "Content to write to the memory."},
                },
                "required": ["label", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_replace",
            "description": "Replace the contents of core memory. To delete memories, use an empty string for new_content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Section of the memory to be edited."},
                    "old_content": {"type": "string", "description": "String to replace. Must be an exact match."},
                    "new_content": {"type": "string", "description": "Content to write to the memory."},
                },
                "required": ["label", "old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_insert",
            "description": (
                "Add information to long-term archival memory for later retrieval. "
                "Store self-contained facts or summaries, not conversational fragments. "
                "Optionally attach category tags to make information easier to find."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to store. Should be clear and self-contained."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of category tags, e.g. ['meetings', 'project-updates'].",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_search",
            "description": (
                "Search archival memory using semantic similarity. Query by concept/meaning, not exact phrases. "
                "Use tags to narrow results when you know the category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you're looking for, described naturally."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter to memories with these tags."},
                    "tag_match_mode": {
                        "type": "string",
                        "enum": ["any", "all"],
                        "description": "'any' = ANY of the tags; 'all' = ALL tags must match.",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max number of results (default 10)."},
                    "start_datetime": {"type": "string", "description": "ISO 8601: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM'."},
                    "end_datetime": {"type": "string", "description": "ISO 8601 (inclusive to end-of-day for date-only)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conversation_search",
            "description": (
                "Search prior conversation history by text. Query tokens are ANDed case-insensitively. "
                "Use for recalling older messages evicted from the in-context FIFO by compaction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search string. Optional if filtering by time/role."},
                    "roles": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["assistant", "user", "tool"]},
                        "description": "Optional list of message roles to filter by.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max results (default 10)."},
                    "start_date": {"type": "string", "description": "ISO 8601: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM'."},
                    "end_date": {"type": "string", "description": "ISO 8601 (inclusive to end-of-day for date-only)."},
                },
                "required": [],
            },
        },
    },
]


async def dispatch(agent: "MemGPTAgent", name: str, args: dict[str, Any]) -> str:
    """Execute a tool call and return a string result to feed back to the LLM."""
    if name == "send_message":
        agent.last_send_message = args["message"]
        return "Sent message successfully."

    if name == "core_memory_append":
        # Mirrors Letta exactly: appends and returns the full updated block value.
        return agent.core_memory.append(args["label"], args["content"])

    if name == "core_memory_replace":
        # Mirrors Letta: returns the full updated block value; on miss, raises ValueError
        # which the agent loop catches and surfaces to the LLM as a tool_result.
        return agent.core_memory.replace(args["label"], args["old_content"], args["new_content"])

    if name == "archival_memory_insert":
        pid = await agent.archival.insert(args["content"], tags=args.get("tags"))
        return f"Stored passage {pid[:8]} (tags={args.get('tags') or []})."

    if name == "archival_memory_search":
        results = await agent.archival.search(
            query=args["query"],
            tags=args.get("tags"),
            tag_match_mode=args.get("tag_match_mode", "any"),
            top_k=args.get("top_k"),
            start_datetime=args.get("start_datetime"),
            end_datetime=args.get("end_datetime"),
        )
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results):
            tag_s = f" tags={r.tags}" if r.tags else ""
            lines.append(f"[{i + 1}] ({r.score:.2f}){tag_s} {r.text}")
        return f"Found {len(results)} results:\n" + "\n".join(lines)

    if name == "conversation_search":
        results = await agent.recall.search(
            query=args.get("query"),
            roles=args.get("roles"),
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            limit=args.get("limit", 10),
        )
        if not results:
            return "No results found."
        lines = []
        for r in results:
            who = f"{r.role}/{r.tool_name}" if r.tool_name else r.role
            lines.append(f"[{r.created_at:%Y-%m-%d %H:%M}] {who}: {r.content[:200]}")
        return f"Found {len(results)} results:\n" + "\n".join(lines)

    return f"Unknown tool: {name}"
