"""
Application configuration — loaded from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Application ---
    app_name: str = "IntentRAG"
    debug: bool = False

    # --- Database (PostgreSQL) ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/intentrag"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_base_url: str = ""  # Leave empty for OpenAI, set for Groq/other providers
    default_llm_model: str = "llama-3.3-70b-versatile"  # Groq model

    # --- Cohere ---
    cohere_api_key: str = ""

    # --- LangFuse ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- JWT ---
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # --- Intent Classifier ---
    intent_model_path: str = ""  # Empty = use default ./intent_classifier_model_roberta
    intent_categories: list[str] = [
        "factual",
        "person",
        "time",
        "location",
        "explanation",
        "other",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
