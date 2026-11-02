"""Runtime configuration, loaded from environment variables (prefix ``RAG_``)."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    chunking: ChunkingSettings = ChunkingSettings()
    contextual_llm: bool = False
