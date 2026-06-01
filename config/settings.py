from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    default_ttl: int = 3600
    embedding_model: str = "text-embedding-3-small"

    threshold_factual: float = 0.96
    threshold_creative: float = 0.90
    threshold_code: float = 0.94
    threshold_default: float = 0.93

    class Config:
        env_file = ".env"


settings = Settings()
