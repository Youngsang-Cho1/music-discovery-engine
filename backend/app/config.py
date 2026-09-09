"""Application settings, loaded from environment / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Service ---
    app_name: str = "Music Discovery Engine"
    cors_origins: str = "http://localhost:3000"

    # --- Postgres (catalog metadata + interaction events) ---
    database_url: str = "postgresql+psycopg://mde:mde@localhost:5432/mde"

    # --- Qdrant (ANN over CLAP audio embeddings) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "tracks"

    # --- Redis (cache + bandit posterior state) ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Jamendo (CC-licensed catalog source) ---
    # Register a free client id at https://devportal.jamendo.com/
    jamendo_client_id: str = ""
    jamendo_api_base: str = "https://api.jamendo.com/v3.0"

    # --- CLAP embedding model ---
    clap_model_id: str = "laion/larger_clap_music_and_speech"
    clap_sample_rate: int = 48_000
    # Seconds of audio fed to CLAP per track. The model was trained on 10s windows.
    clap_window_seconds: int = 10
    embedding_dim: int = 512

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
