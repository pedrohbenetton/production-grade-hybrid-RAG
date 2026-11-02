"""Runtime configuration, loaded from environment variables (prefix ``RAG_``)."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_tokens: int = Field(64, ge=1, description="Chunks below this are merged with neighbours.")
    max_tokens: int = Field(512, ge=8, description="Hard cap, except for atomic code/table blocks.")
    breakpoint_percentile: float = Field(
        90.0, gt=0, lt=100, description="Semantic split where distance exceeds this percentile."
    )
    window: int = Field(1, ge=0, description="Neighbour sentences embedded with each sentence.")

    @model_validator(mode="after")
    def _min_below_max(self) -> Self:
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")
        return self


class EmbeddingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = Field(64, ge=1)


class QdrantSettings(BaseModel):
    """``url`` → server mode; else ``path`` → embedded on-disk; else in-memory."""

    model_config = ConfigDict(frozen=True)

    url: str | None = None
    path: str | None = "./data/qdrant"
    api_key: SecretStr | None = None
    collection: str = "chunks"


class BM25Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    k1: float = Field(1.2, ge=0)
    b: float = Field(0.75, ge=0, le=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    chunking: ChunkingSettings = ChunkingSettings()
    contextual_llm: bool = False
    embedding: EmbeddingSettings = EmbeddingSettings()
    qdrant: QdrantSettings = QdrantSettings()
    bm25: BM25Settings = BM25Settings()
