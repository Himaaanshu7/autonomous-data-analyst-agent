import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = BASE_DIR / ".env"


def _bootstrap_env() -> None:
    """Load env vars in correct priority order:
      1. .env file  (local dev — highest local priority)
      2. Streamlit secrets (cloud deploy — fills any keys still missing)

    Both are injected into os.environ so pydantic-settings sees them.
    os.environ values set before this call (e.g. real shell exports) are
    never overwritten.
    """
    # Step 1: .env — load keys not already in the real environment
    try:
        from dotenv import dotenv_values
        for key, value in dotenv_values(str(_ENV_FILE)).items():
            if value and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass

    # Step 2: Streamlit secrets — fill any keys still absent
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_bootstrap_env()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM  (Groq — free tier)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    llm_model: str = Field(default="llama-3.3-70b-versatile", alias="LLM_MODEL")
    llm_max_tokens: int = 4096

    # Database
    duckdb_path: str = Field(
        default=str(BASE_DIR / "data" / "db" / "analytics.duckdb"),
        alias="DUCKDB_PATH",
    )

    # Paths
    sample_data_dir: str = str(BASE_DIR / "data" / "sample")

    # Agent behaviour
    max_retries: int = 3
    anomaly_zscore_threshold: float = 3.0
    anomaly_iqr_multiplier: float = 1.5


settings = Settings()
