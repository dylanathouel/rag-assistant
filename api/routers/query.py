"""
POST /query — Lance une requête RAG.

Workflow :
  1. Reçoit { question, top_k }
  2. search_chunks() → top_k chunks pertinents
  3. generate_answer() → réponse Claude
  4. Renvoie { answer, sources }
"""
from fastapi import APIRouter

from api.schemas import QueryRequest, QueryResponse, Source
from query import search_chunks, generate_answer

router = APIRouter()


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    chunks = search_chunks(req.question, top_k=req.top_k)

    if not chunks:
        return QueryResponse(
            question=req.question,
            answer="Je ne sais pas — aucun contexte pertinent trouvé en base.",
            sources=[],
        )

    answer = generate_answer(req.question, chunks)

    sources = [
        Source(
            source=src,
            title=title,
            similarity=float(sim),
            content_preview=content[:200] + ("..." if len(content) > 200 else ""),
        )
        for content, src, title, sim in chunks
    ]

    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=sources,
    )