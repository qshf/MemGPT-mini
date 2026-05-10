import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import and_, case, func, literal, or_, select

from memgpt.db import Database, MessageRow


@dataclass
class RecallResult:
    id: str
    role: str
    content: str
    tool_name: str | None
    created_at: datetime


class RecallMemory:
    """Full conversation log. Auto-persisted; searchable by text."""

    def __init__(self, db: Database, agent_id: str):
        self.db = db
        self.agent_id = agent_id

    async def append(self, msg: dict[str, Any], tool_name: str | None = None) -> str:
        """Persist an OpenAI-style message dict. `tool_name` is a debug-only
        column populated for role=tool rows so DB readers can see which tool
        produced the result — it's never sent back to the LLM."""
        row = MessageRow(
            agent_id=self.agent_id,
            role=msg.get("role", "user"),
            content=msg.get("content") or "",
            tool_call_id=msg.get("tool_call_id"),
            tool_name=tool_name,
            tool_calls_json=json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None,
        )
        async with self.db.session() as s:
            s.add(row)
            await s.commit()
            await s.refresh(row)
        return row.id

    async def search(
        self,
        query: str | None = None,
        roles: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> list[RecallResult]:
        """Text search. Query is split on whitespace; tokens are ANDed (case-insensitive).
        start_date / end_date accept ISO 8601: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM".
        Date-only end_date includes the full day."""
        from datetime import datetime, time

        def _parse(d: str, end: bool) -> datetime:
            if "T" in d:
                return datetime.fromisoformat(d)
            base = datetime.fromisoformat(d)
            return datetime.combine(base.date(), time.max if end else time.min)

        async with self.db.session() as s:
            stmt = select(MessageRow)
            score = None
            if query and query.strip():
                tokens = [t for t in query.split() if t]
                if tokens:
                    # Hybrid-style: OR across tokens, rank by #tokens matched, then recency.
                    score = sum(
                        (case((MessageRow.content.ilike(f"%{t}%"), 1), else_=0) for t in tokens),
                        literal(0),
                    ).label("score")
                    stmt = stmt.add_columns(score)
                    stmt = stmt.where(or_(*(MessageRow.content.ilike(f"%{t}%") for t in tokens)))
            stmt = stmt.where(MessageRow.agent_id == self.agent_id)
            if roles:
                stmt = stmt.where(MessageRow.role.in_(roles))
            if start_date:
                stmt = stmt.where(MessageRow.created_at >= _parse(start_date, end=False))
            if end_date:
                stmt = stmt.where(MessageRow.created_at <= _parse(end_date, end=True))
            if score is not None:
                stmt = stmt.order_by(score.desc(), MessageRow.created_at.desc())
            else:
                stmt = stmt.order_by(MessageRow.created_at.desc())
            stmt = stmt.limit(limit)
            result = (await s.execute(stmt)).all()
            rows = [row[0] for row in result]
        return [
            RecallResult(id=r.id, role=r.role, content=r.content, tool_name=r.tool_name, created_at=r.created_at)
            for r in rows
        ]

    async def size(self) -> int:
        from sqlalchemy import func
        async with self.db.session() as s:
            stmt = select(func.count(MessageRow.id)).where(MessageRow.agent_id == self.agent_id)
            return (await s.execute(stmt)).scalar_one()
