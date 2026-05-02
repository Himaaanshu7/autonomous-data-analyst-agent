import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_streamlit_secrets() -> None:
    """Inject Streamlit secrets into environment variables before settings load.

    This makes the same Settings class work for both local (.env) and
    Streamlit Cloud (secrets.toml) without any change to calling code.
    """
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass  # Not running inside Streamlit, or secrets not configured


_load_streamlit_secrets()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="claude-sonnet-4-6", alias="LLM_MODEL")
    llm_max_tokens: int = 4096

    # Database
    duckdb_path: str = Field(
        default=str(BASE_DIR / "data" / "db" / "analytics.duckdb"),
        alias="DUCKDB_PATH",
    )
    postgres_url: str = Field(default="", alias="POSTGRES_URL")

    # Paths
    sample_data_dir: str = str(BASE_DIR / "data" / "sample")

    # Agent behaviour
    max_retries: int = 3
    anomaly_zscore_threshold: float = 3.0
    anomaly_iqr_multiplier: float = 1.5


settings = Settings()
