from pathlib import Path
import os 
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR/ ".env")


class Settings(BaseSettings):
    """Centralized configuration loaded from .env file."""

    # API Keys
    api_key: str = os.getenv("API_KEY")
    llamaparse_api_key: str = os.getenv("LLAMAPARSE_API_KEY")

    # LLM
    llm_endpoint: str = os.getenv("ENDPOINT")
    llm_model_id: str = os.getenv("MODEL_ID")
    llm_temperature: float = 0

    # Embeddings
    embedding_endpoint: str = os.getenv("OLLAMA_ENDPOINT")
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL")

    # Re-Ranking
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME")

    # Paths
    chroma_path: str = str(BASE_DIR / "vectorstore")
    data_dir: str = str(BASE_DIR / "data")

    # Retriever
    retriever_k: int = 4

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
