"""Centralized configuration via Pydantic BaseSettings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    model_config = {"env_prefix": "LLM_"}

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.0


class ChromaSettings(BaseSettings):
    model_config = {"env_prefix": "CHROMA_"}

    host: str = "localhost"
    port: int = 8001
    collection_name: str = "aars_documents"


class EmbeddingSettings(BaseSettings):
    model_config = {"env_prefix": "EMBEDDING_"}

    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64


class RetrieverSettings(BaseSettings):
    model_config = {"env_prefix": "RETRIEVER_"}

    top_k: int = Field(default=10, alias="TOP_K")
    bm25_top_k: int = 10
    graph_max_hops: int = 2
    graph_top_k: int = 10


class FusionSettings(BaseSettings):
    model_config = {"env_prefix": "FUSION_"}

    rrf_k: int = Field(default=60, alias="RRF_K")
    mmr_lambda: float = Field(default=0.5, alias="MMR_LAMBDA")
    final_top_k: int = 5


class ChunkerSettings(BaseSettings):
    model_config = {"env_prefix": "CHUNK_"}

    size: int = Field(default=512, alias="CHUNK_SIZE")
    overlap: int = Field(default=64, alias="CHUNK_OVERLAP")


class RerankerSettings(BaseSettings):
    model_config = {"env_prefix": "RERANKER_"}

    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enabled: bool = True


class HallucinationSettings(BaseSettings):
    model_config = {"env_prefix": "HALLUCINATION_"}

    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    mode: str = "llm"
    threshold: float = 0.5


class PipelineSettings(BaseSettings):
    model_config = {"env_prefix": "PIPELINE_"}

    max_reflection_iterations: int = Field(
        default=3, alias="MAX_REFLECTION_ITERATIONS"
    )
    trace_enabled: bool = True


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    llm: LLMSettings = LLMSettings()
    chroma: ChromaSettings = ChromaSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retriever: RetrieverSettings = RetrieverSettings()
    fusion: FusionSettings = FusionSettings()
    chunker: ChunkerSettings = ChunkerSettings()
    pipeline: PipelineSettings = PipelineSettings()
    reranker: RerankerSettings = RerankerSettings()
    hallucination: HallucinationSettings = HallucinationSettings()


def get_settings() -> Settings:
    return Settings()
