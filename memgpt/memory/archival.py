from dataclasses import dataclass
from datetime import datetime, time
from typing import Literal

from openai import AsyncOpenAI
from sqlalchemy import and_, or_, select

from memgpt.config import Config
from memgpt.db import Database, Passage


@dataclass
class PassageResult:
    id: str
    text: str
    tags: list[str] | None
    created_at: datetime
    score: float


def _parse_iso(d: str, *, end: bool) -> datetime:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM'. For date-only + end, expand to 23:59:59."""
    if "T" in d:
        return datetime.fromisoformat(d)
    base = datetime.fromisoformat(d)
    return datetime.combine(base.date(), time.max if end else time.min)


class ArchivalMemory:
    """Long-term store scoped to a single ``archive_id``.

    Uses pgvector semantic search when an embedding provider is configured;
    otherwise falls back to ILIKE text search.

    Matches Letta's model where each agent has a private archive by default
    (``archive_id == agent_id``); multiple agents can share an archive by
    passing the same id at construction time."""

    def __init__(self, cfg: Config, db: Database, archive_id: str):
        self.cfg = cfg
        self.db = db
        self.archive_id = archive_id
        self.embed_client: AsyncOpenAI | None = None
        if cfg.embed_model and cfg.embed_api_key:
            self.embed_client = AsyncOpenAI(api_key=cfg.embed_api_key, base_url=cfg.embed_base_url)

    async def _embed(self, text: str) -> list[float] | None:
        if not self.embed_client:
            return None
        kwargs: dict = {"model": self.cfg.embed_model, "input": text}
        if self.cfg.embed_base_url and "dashscope" in self.cfg.embed_base_url:
            kwargs["dimensions"] = self.cfg.embed_dim
            kwargs["encoding_format"] = "float"
        resp = await self.embed_client.embeddings.create(**kwargs)
        return resp.data[0].embedding

    async def insert(self, text: str, tags: list[str] | None = None) -> str:
        embedding = await self._embed(text)
        p = Passage(archive_id=self.archive_id, text=text, embedding=embedding, tags=tags)
        async with self.db.session() as s:
            s.add(p)
            await s.commit()
            await s.refresh(p)
        return p.id

    async def search(
        self,
        query: str,
        tags: list[str] | None = None,
        tag_match_mode: Literal["any", "all"] = "any",
        top_k: int | None = None,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
    ) -> list[PassageResult]:
        top_k = top_k or self.cfg.retrieval_page_size
        q_emb = await self._embed(query)

        filters = [Passage.archive_id == self.archive_id]
        if tags:
            if tag_match_mode == "all":
                filters.append(Passage.tags.contains(tags))  # type: ignore[arg-type]
            else:
                filters.append(Passage.tags.overlap(tags))  # type: ignore[arg-type]
        if start_datetime:
            filters.append(Passage.created_at >= _parse_iso(start_datetime, end=False))
        if end_datetime:
            filters.append(Passage.created_at <= _parse_iso(end_datetime, end=True))

        async with self.db.session() as s:
            if q_emb is not None:
                distance = Passage.embedding.cosine_distance(q_emb).label("distance")
                stmt = select(Passage, distance).where(Passage.embedding.is_not(None))
                if filters:
                    stmt = stmt.where(and_(*filters))
                stmt = stmt.order_by(distance).limit(top_k)
                rows = (await s.execute(stmt)).all()
                return [
                    PassageResult(id=p.id, text=p.text, tags=p.tags, created_at=p.created_at, score=1 - d)
                    for p, d in rows
                ]
            stmt = select(Passage).where(Passage.text.ilike(f"%{query}%"))
            if filters:
                stmt = stmt.where(and_(*filters))
            stmt = stmt.order_by(Passage.created_at.desc()).limit(top_k)
            rows = (await s.execute(stmt)).scalars().all()
            return [
                PassageResult(id=p.id, text=p.text, tags=p.tags, created_at=p.created_at, score=0.0)
                for p in rows
            ]
