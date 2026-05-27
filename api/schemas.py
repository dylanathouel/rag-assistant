"""
api/schemas.py — Modèles Pydantic pour les requêtes/réponses de l'API.

Pourquoi Pydantic :
- Validation automatique des inputs (type, longueur, format...) → réponses 422 propres si KO
- Sérialisation auto vers JSON pour les outputs
- Génère la doc OpenAPI à /docs — un client TS/Python peut générer son SDK automatiquement
"""
from pydantic import BaseModel, Field


# ===== Query =====

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="La question utilisateur")
    top_k: int = Field(5, ge=1, le=20, description="Nombre de chunks à récupérer")
    hybrid: bool = Field(True, description="Hybrid search BM25+vecteur (True) ou vectoriel pur (False)")


class Source(BaseModel):
    source: str
    title: str | None = None
    similarity: float
    content_preview: str  # 200 premiers caractères, pour pas saturer la response


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


# ===== Ingest =====

class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="URL à télécharger et ingérer")
    source: str = Field(..., description="Identifiant unique du document (souvent l'URL canonique)")
    title: str | None = None
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    source: str
    document_id: int
    chunks_count: int