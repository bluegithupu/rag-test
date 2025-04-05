"""
Configuration settings for the RAG system.
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Settings for the RAG system."""

    # OpenAI API Key
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = Field(default=os.getenv("OPENAI_BASE_URL", ""))

    # Vector DB settings
    vector_db_path: str = Field(default=os.getenv("VECTOR_DB_PATH", "./data/vectordb"))

    # LLM settings
    llm_model: str = Field(default=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    llm_temperature: float = Field(default=float(os.getenv("LLM_TEMPERATURE", "0.0")))

    # Embedding settings
    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))

    # Document processing settings
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "200")))

    # Retrieval settings
    num_documents: int = Field(default=int(os.getenv("NUM_DOCUMENTS", "4")))

    # Logging settings
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "info"))
    log_to_console: bool = Field(default=os.getenv("LOG_TO_CONSOLE", "true").lower() == "true")
    log_to_file: bool = Field(default=os.getenv("LOG_TO_FILE", "true").lower() == "true")

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()
