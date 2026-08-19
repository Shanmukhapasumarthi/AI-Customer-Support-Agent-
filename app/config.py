from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# PROJECT_ROOT is the directory that contains this project (one level above
# `app/`). We resolve paths against it so the app behaves identically whether
# you run it from the project root, from `frontend/`, or from Docker.
#   __file__            -> <root>/app/config.py
#   .resolve().parent   -> <root>/app
#   .parent             -> <root>
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed, validated application settings.

    `BaseSettings` is Pydantic's configuration class. On instantiation it
    reads each field from (in priority order): an explicit keyword argument,
    a real OS environment variable, then the `.env` file. Whatever it finds
    as a string is COERCED into the annotated Python type, and if coercion
    fails you get a loud error at startup instead of a mysterious crash
    three modules deep.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Ignore any extra variables in .env we haven't declared here, so a
        # stray line doesn't crash the whole app.
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    # `...` (Ellipsis) as the default means REQUIRED. If GROQ_API_KEY is
    # missing, the app refuses to start -- which is what we want, because a
    # missing key would otherwise fail on the first user question instead.
    groq_api_key: str = Field(..., description="Groq API key from console.groq.com")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model id. MUST support tool calling.",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # --- Retrieval ---------------------------------------------------------
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    retriever_top_k: int = Field(default=4, gt=0)

    # --- Paths (relative strings from .env; turned absolute below) ---------
    knowledge_base_dir: str = Field(default="data/knowledge_base")
    vector_store_dir: str = Field(default="data/vector_store")
    database_url: str = Field(default="sqlite:///database/customer_support.db")

    # --- App ---------------------------------------------------------------
    app_name: str = Field(default="AI Customer Support Knowledge Agent")
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # -- Validators ---------------------------------------------------------
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, v: int, info) -> int:
        """Catch a classic RAG misconfiguration before it wastes your time.

        If overlap >= chunk_size, the text splitter can loop forever or emit
        near-duplicate chunks. Better to fail here with a clear message.
        """
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({v}) must be smaller than CHUNK_SIZE ({chunk_size})"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def log_level_must_be_valid(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(valid)}, got {v!r}")
        return upper

    # -- Convenience properties --------------------------------------------
    # These turn the relative strings from .env into absolute Path objects.
    # Doing it here means no other module ever has to think about cwd.
    @property
    def knowledge_base_path(self) -> Path:
        return PROJECT_ROOT / self.knowledge_base_dir

    @property
    def vector_store_path(self) -> Path:
        return PROJECT_ROOT / self.vector_store_dir

    @property
    def sqlite_path(self) -> Path:
        """Strip the `sqlite:///` prefix to get a plain filesystem path."""
        raw = self.database_url.replace("sqlite:///", "")
        return PROJECT_ROOT / raw


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    `@lru_cache` means the `.env` file is parsed exactly once per process, no
    matter how many modules call `get_settings()`. Every caller gets the same
    object back. This is the standard FastAPI dependency-injection pattern and
    it also makes tests easy: call `get_settings.cache_clear()` to force a
    reload with different env vars.
    """
    return Settings()  # type: ignore[call-arg]


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure root logging once, and return this app's logger.

    WHY LOGGING AND NOT print()?
    An agent makes non-obvious decisions (which tool did it pick? what did the
    retriever return?). `print` gives you no levels, no timestamps, and no way
    to turn it off in production. Logging gives you all three for free.
    """
    settings = get_settings()
    logging.basicConfig(
        level=level or settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries are extremely chatty at INFO. Mute them.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "groq", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("support_agent")


# Module-level singleton, imported everywhere as:
#     from app.config import settings
settings = get_settings()
