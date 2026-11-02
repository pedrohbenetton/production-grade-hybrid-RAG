"""Runtime configuration, loaded from environment variables (prefix ``RAG_``)."""

from __future__ import annotations

from typing import Literal, Self

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
    cache_dir: str | None = "./data/models"  # FastEmbed defaults to a volatile temp dir


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


class RetrievalSettings(BaseModel):
    """Candidate funnel (§4.3): k_d / k_s per retriever → fused top-N → reranked final_k."""

    model_config = ConfigDict(frozen=True)

    dense_candidates: int = Field(50, ge=1)
    sparse_candidates: int = Field(50, ge=1)
    fused_candidates: int = Field(30, ge=1)
    final_k: int = Field(5, ge=1, le=50)
    rrf_k: int = Field(60, ge=0)
    dense_weight: float = Field(1.0, ge=0)
    sparse_weight: float = Field(1.0, ge=0)


class RerankSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["flashrank", "cohere", "none"] = "flashrank"
    model: str | None = None  # provider default when unset
    cohere_api_key: SecretStr | None = None
    cache_dir: str = "./data/models"


class LLMSettings(BaseModel):
    """Generator. Without a key (here or ``ANTHROPIC_API_KEY``) /v1/query is disabled."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["anthropic", "none"] = "anthropic"
    model: str = "claude-sonnet-5-5"
    api_key: SecretStr | None = None
    max_tokens: int = Field(1024, ge=1)
    temperature: float | None = Field(None, ge=0, le=1)  # unset → provider default
    context_token_budget: int = Field(3000, ge=1)


class GuardrailSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    nli_repo: str = "cross-encoder/nli-deberta-v3-small"
    nli_onnx_file: str = "onnx/model_quint8_avx2.onnx"  # int8; use onnx/model.onnx off x86
    nli_cache_dir: str = "./data/models"
    entailment_threshold: float = Field(0.5, gt=0, le=1)
    policy: Literal["annotate", "filter", "abstain"] = "filter"
    min_faithfulness: float = Field(0.8, ge=0, le=1)  # used by policy "abstain"
    # Abstain before generating when the top hit scores below this. Scores are only
    # comparable within a mode: it is meaningful for reranker scores, not RRF scores.
    min_relevance: float | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )

    chunking: ChunkingSettings = ChunkingSettings()
    contextual_llm: bool = False
    embedding: EmbeddingSettings = EmbeddingSettings()
    qdrant: QdrantSettings = QdrantSettings()
    bm25: BM25Settings = BM25Settings()
    retrieval: RetrievalSettings = RetrievalSettings()
    rerank: RerankSettings = RerankSettings()
    llm: LLMSettings = LLMSettings()
    guardrails: GuardrailSettings = GuardrailSettings()
