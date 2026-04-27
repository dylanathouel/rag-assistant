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

# ===== POSTGRES =====
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "macdedylan")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Connexion string Postgres
DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ===== RAG Parameters =====
CHUNK_SIZE = 1000  # Caractères par chunk
CHUNK_OVERLAP = 200  # Chevauchement entre chunks
TOP_K = 5  # Nombre de chunks à récupérer pour chaque query
