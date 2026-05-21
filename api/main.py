"""
api/main.py — App FastAPI principale.
Lance avec : uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI

from api.routers import health, query, ingest

app = FastAPI(
    title="RAG Assistant",
    description="RAG sur corpus de documents techniques. Foundation pour la roadmap IA 2026.",
    version="0.1.0",
)

# Chaque router est monté avec un préfixe.
# tags = regroupement dans la doc OpenAPI /docs.
app.include_router(health.router, tags=["health"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])


@app.get("/")
def root():
    return {
        "name": "RAG Assistant",
        "version": "0.1.0",
        "docs": "/docs",
    }