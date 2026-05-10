import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Chat LLM (DeepSeek, OpenAI, anything OpenAI-compatible)
    chat_api_key: str
    chat_base_url: str | None
    model: str
    reasoning_effort: str | None = None
    enable_thinking: bool = False

    # Embedding provider (optional — DeepSeek has no embeddings API).
    # If unset, archival memory falls back to ILIKE text search.
    embed_api_key: str | None = None
    embed_base_url: str | None = None
    embed_model: str | None = None
    embed_dim: int = 1024

    # Infra
    pg_uri: str = "postgresql+asyncpg://memgpt:memgpt@localhost:5432/memgpt_mini"
    context_window: int = 128000

    # Compaction
    summarization_trigger: float = 0.9
    summarization_mode: str = "sliding_window"  # "sliding_window" | "all"
    sliding_window_percentage: float = 0.3  # fraction of messages to evict per pass; Letta default
    summarizer_clip_chars: int = 50000
    include_summary_ack: bool = False
    message_buffer_min: int = 3
    core_memory_char_limit: int = 2000
    retrieval_page_size: int = 10


def load_config() -> Config:
    chat_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MEMGPT_CHAT_API_KEY")
    if not chat_key:
        raise RuntimeError("DEEPSEEK_API_KEY (or MEMGPT_CHAT_API_KEY) is not set")

    embed_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MEMGPT_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY")

    return Config(
        chat_api_key=chat_key,
        chat_base_url=os.environ.get("MEMGPT_CHAT_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("MEMGPT_MODEL", "deepseek-v4-pro"),
        reasoning_effort=os.environ.get("MEMGPT_REASONING_EFFORT") or None,
        enable_thinking=os.environ.get("MEMGPT_THINKING", "0") not in ("", "0", "false", "False"),
        embed_api_key=embed_key,
        embed_base_url=os.environ.get("MEMGPT_EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1") if embed_key else None,
        embed_model=os.environ.get("MEMGPT_EMBED_MODEL", "text-embedding-v3") if embed_key else None,
        embed_dim=int(os.environ.get("MEMGPT_EMBED_DIM", "1024")),
        pg_uri=os.environ.get("MEMGPT_PG_URI", "postgresql+asyncpg://memgpt:memgpt@localhost:5432/memgpt_mini"),
        context_window=int(os.environ.get("MEMGPT_CONTEXT_WINDOW", "128000")),
        summarization_mode=os.environ.get("MEMGPT_SUMMARIZATION_MODE", "sliding_window"),
        sliding_window_percentage=float(os.environ.get("MEMGPT_SLIDING_WINDOW_PCT", "0.3")),
    )


def chat_extras(cfg: Config) -> dict:
    """Build kwargs for chat.completions.create that are DeepSeek/reasoning-specific."""
    extras: dict = {}
    if cfg.reasoning_effort:
        extras["reasoning_effort"] = cfg.reasoning_effort
    if cfg.enable_thinking:
        extras["extra_body"] = {"thinking": {"type": "enabled"}}
    return extras
