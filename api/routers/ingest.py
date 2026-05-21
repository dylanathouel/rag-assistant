"""
POST /ingest/url — Ingest un document depuis une URL.
"""
from fastapi import APIRouter, HTTPException
import psycopg

from api.schemas import IngestUrlRequest, IngestResponse
from config import DB_CONNECTION_STRING
from ingest import ingest_url

router = APIRouter()


@router.post("/url", response_model=IngestResponse)
def ingest_from_url(req: IngestUrlRequest) -> IngestResponse:
    try:
        ingest_url(req.url, req.source, req.title, req.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    # Récupère le doc_id et compte les chunks ingérés
    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id, count(c.id)
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.source = %s
                GROUP BY d.id
                """,
                (req.source,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=500, detail="Ingéré mais introuvable en DB")

    return IngestResponse(
        source=req.source,
        document_id=row[0],
        chunks_count=row[1],
    )