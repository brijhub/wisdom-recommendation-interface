"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    chroma_path: Path = BASE_DIR / "storage" / "chroma_db"
    chroma_collection: str = "wisdom_knowledge"
    backend_url: str = "http://127.0.0.1:8000"
    retrieval_limit: int = 6

    def require_openai_key(self) -> None:
        if not self.openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env and try again.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

