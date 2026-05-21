"""
GET /health — Vérifie que l'API + Postgres répondent, et donne un état du corpus.
Utile pour Uptime Kuma / Langfuse en S6, et pour Docker healthcheck.
"""
from fastapi import APIRouter
import psycopg

from config import DB_CONNECTION_STRING

router = APIRouter()


@router.get("/health")
def health():
    try:
        with psycopg.connect(DB_CONNECTION_STRING, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM documents")
                doc_count = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM chunks")
                chunk_count = cur.fetchone()[0]
        return {
            "status": "ok",
            "documents": doc_count,
            "chunks": chunk_count,
        }
    except Exception as e:
        # 200 OK quand même mais avec status=error, pour qu'un monitoring puisse lire le body
        return {"status": "error", "detail": str(e)}