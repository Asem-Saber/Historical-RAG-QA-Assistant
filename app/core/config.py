from pathlib import Path
import os 
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR/ ".env")


class Settings(BaseSettings):
    """Centralized configuration loaded from .env file."""

    # API Keys
    github_api_key: str = os.getenv("GITHUB_API_KEY")
    llamaparse_api_key: str = os.getenv("LLAMAPARSE_API_KEY")

    # LLM
    llm_endpoint: str = "https://models.github.ai/inference"
    llm_model_id: str = "openai/gpt-4.1"
    llm_temperature: float = 0

    # Embeddings
    embedding_model: str = "sentence-transformers/LaBSE"
    embedding_device: str = "cuda"

    # Paths
    chroma_path: str = str(BASE_DIR / "vectorstore")
    data_dir: str = str(BASE_DIR / "data")

    # Retriever
    retriever_k: int = 5

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
