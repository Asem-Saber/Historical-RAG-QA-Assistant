from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings(BaseSettings):
    """Centralized configuration loaded from .env file."""

    # API Keys
    api_key: str
    llamaparse_api_key: str = ""

    # LLM
    llm_endpoint: str = Field(
        validation_alias=AliasChoices("llm_endpoint", "endpoint"),
    )
    llm_model_id: str = Field(
        validation_alias=AliasChoices("llm_model_id", "model_id"),
    )
    llm_temperature: float = 0

    # Embeddings
    embedding_endpoint: str = Field(
        validation_alias=AliasChoices("embedding_endpoint", "ollama_endpoint"),
    )
    embedding_model: str = Field(
        validation_alias=AliasChoices("embedding_model", "ollama_embedding_model"),
    )

    # Re-Ranking
    reranker_model_name: str

    # Paths
    base_dir: str = str(BASE_DIR)
    data_dir: str = str(BASE_DIR / "data")
    chroma_path: str = str(BASE_DIR / "data" / "vectorstore")

    # CORS
    allowed_origins: list[str] = ["http://localhost:8501", "http://localhost:8000"]

    # Retriever
    retriever_k: int = 4

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
