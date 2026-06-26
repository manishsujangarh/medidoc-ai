from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str
    pinecone_api_key: str
    pinecone_index: str = "medidoc"
    embedding_model: str = "gemini-embedding-2"  # <-- Update this line
    llm_model: str = "gemini-2.5-flash"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    class Config:
        env_file = ".env"

settings = Settings()