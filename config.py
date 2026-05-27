import os
from dotenv import load_dotenv

# Charge les variables du .env
load_dotenv()

# ===== VOYAGE (Embeddings) =====
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = "voyage-3"

# ===== OPENROUTER (LLM pour générer les réponses) =====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4.6")

# ===== ANTHROPIC (SDK direct — Contextual Retrieval avec prompt caching) =====
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CONTEXT_MODEL = os.getenv("CONTEXT_MODEL", "claude-haiku-4-5-20251001")
CONTEXTUAL_RETRIEVAL = os.getenv("CONTEXTUAL_RETRIEVAL", "false").lower() == "true"

# ===== POSTGRES =====
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "macdedylan")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ===== RAG Parameters =====
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# ===== RERANKER =====
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "none")  # "none" | "cohere"
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "30"))   # candidats envoyés au reranker
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", str(TOP_K)))  # résultat final

# ===== COHERE (optionnel — uniquement si RERANKER_BACKEND=cohere) =====
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ===== HYDE =====
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "false").lower() == "true"
