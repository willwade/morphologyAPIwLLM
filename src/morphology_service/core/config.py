from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SourcePreference = Literal["llm", "rule", "hybrid"]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="MORPHOLOGY_", env_file=".env", env_file_encoding="utf-8")

    api_title: str = Field(default="Morphology API")
    gemini_api_key: Optional[str] = Field(default=None, description="API key for Google Gemini")
    gemini_api_key_file: Optional[str] = Field(default=None, description="Path to a file containing the Gemini API key")
    default_source_preference: SourcePreference = "hybrid"
    llm_model_name: str = Field(default="gemini-2.0-flash")
    llm_temperature: float = Field(default=0.0)
    max_llm_retries: int = Field(default=2, ge=0, le=5)
    cache_ttl_seconds: int = Field(default=1800)
    hybrid_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def load_key_from_file(self) -> "Settings":
        if not self.gemini_api_key and self.gemini_api_key_file:
            key_path = Path(self.gemini_api_key_file).expanduser()
            if key_path.is_file():
                self.gemini_api_key = key_path.read_text(encoding="utf-8").strip()
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
