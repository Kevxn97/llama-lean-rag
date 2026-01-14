import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/pdf_rag")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMS: int = int(os.getenv("EMBEDDING_DIMS", "1536"))

    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    TOPK_VEC: int = int(os.getenv("TOPK_VEC", "20"))
    FINAL_EVIDENCE: int = int(os.getenv("FINAL_EVIDENCE", "8"))

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50


config = Config()
