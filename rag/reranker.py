"""
rag/reranker.py — Reranking layer : top-N du hybrid search → top-K pour le prompt.

Backend sélectionné via RERANKER_BACKEND (config.py / .env) :
  "none"   — passthrough, retourne chunks[:top_k] (défaut, pas de dépendance)
  "cohere" — Cohere Rerank v3.5 API  (pip install cohere, COHERE_API_KEY requis)
"""
from config import RERANKER_BACKEND, COHERE_API_KEY


def rerank(question: str, chunks: list[tuple], top_k: int) -> list[tuple]:
    """
    chunks : list of (content, source, title, score)
    Retourne top_k chunks re-scorés, triés par pertinence décroissante.
    """
    if RERANKER_BACKEND == "cohere":
        return _rerank_cohere(question, chunks, top_k)
    return chunks[:top_k]


def _rerank_cohere(question: str, chunks: list[tuple], top_k: int) -> list[tuple]:
    import cohere
    co = cohere.Client(api_key=COHERE_API_KEY)

    documents = [c[0] for c in chunks]
    response = co.rerank(
        model="rerank-v3.5",
        query=question,
        documents=documents,
        top_n=top_k,
    )
    return [
        (
            chunks[r.index][0],
            chunks[r.index][1],
            chunks[r.index][2],
            r.relevance_score,
        )
        for r in response.results
    ]
