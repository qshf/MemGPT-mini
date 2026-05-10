import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memgpt.config import Config


class Base(DeclarativeBase):
    pass


def _uuid_str() -> str:
    return str(uuid.uuid4())


_VECTOR_DIM = 1024  # mutable default; overridden in Database.__init__ before create_all


class Passage(Base):
    """Archival memory entry: text + vector embedding + optional tags.

    Scoped to an ``archive_id`` (mirrors Letta's Archive model). By default each
    agent has a private archive with ``archive_id == agent_id``; multiple agents
    can share an archive by passing the same id."""

    __tablename__ = "passages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    archive_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_VECTOR_DIM), nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MessageRow(Base):
    """Recall memory entry: full conversation log."""

    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    tool_call_id: Mapped[str] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=True)
    tool_calls_json: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


Index("ix_messages_agent_created", MessageRow.agent_id, MessageRow.created_at)


class Database:
    def __init__(self, cfg: Config):
        # Rebind the embedding column to the configured dim before any create_all.
        Passage.__table__.c.embedding.type = Vector(cfg.embed_dim)
        self.cfg = cfg
        self.engine: AsyncEngine = create_async_engine(cfg.pg_uri, echo=False)
        self.session_maker = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
            # Additive migrations: safe to re-run on an existing DB.
            await conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS tool_name VARCHAR(64)"))
            await conn.execute(text("ALTER TABLE passages ADD COLUMN IF NOT EXISTS tags TEXT[]"))
            await conn.execute(text("ALTER TABLE passages ADD COLUMN IF NOT EXISTS archive_id VARCHAR(64)"))
            # Backfill a default archive_id for pre-existing rows (scoped to a legacy archive
            # so new per-agent archives don't accidentally surface them).
            await conn.execute(text("UPDATE passages SET archive_id = 'legacy-shared' WHERE archive_id IS NULL"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_passages_archive_id ON passages (archive_id)"))
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_passages_embedding_hnsw "
                    "ON passages USING hnsw (embedding vector_cosine_ops)"
                )
            )

    def session(self) -> AsyncSession:
        return self.session_maker()
