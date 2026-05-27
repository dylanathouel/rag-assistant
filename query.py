"""
query.py — Recherche des chunks pertinents et génère une réponse.

Pipeline complet (toutes options actives) :
  1. [HyDE]     Génère une réponse hypothétique pour améliorer l'embedding
  2. [Hybrid]   Vector (HNSW) + BM25 (tsvector) fusionnés via RRF → top-N
  3. [Reranker] Cross-encoder (Cohere) top-N → top-K
  4. [LLM]      generate_answer() → Claude via OpenRouter
"""
import requests
import psycopg

from config import (
    DB_CONNECTION_STRING, OPENROUTER_API_KEY, LLM_MODEL, TOP_K,
    RERANKER_BACKEND, RERANKER_TOP_N, HYDE_ENABLED,
)
from utils import get_embedding


def search_chunks_hybrid(embed_text: str, question_text: str, top_k: int) -> list[tuple]:
    """
    Hybrid search : vecteur (HNSW) + BM25 (tsvector), fusionnés par RRF.

    embed_text    : texte à embedder (question brute ou paragraphe HyDE)
    question_text : question originale utilisée côté BM25 (preserve les mots-clés)

    RRF score = 1/(60 + rang_vecteur) + 1/(60 + rang_BM25)
    Le CTE vector_ranked utilise ORDER BY <=> directement → index HNSW actif.
    """
    embedding = get_embedding(embed_text)

    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH vector_ranked AS (
                    SELECT
                        c.id,
                        ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS rank
                    FROM chunks c
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT 60
                ),
                text_ranked AS (
                    SELECT
                        c.id,
                        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, query) DESC) AS rank
                    FROM chunks c, plainto_tsquery('english', %s) AS query
                    WHERE c.tsv @@ query
                    LIMIT 60
                ),
                rrf AS (
                    SELECT
                        COALESCE(v.id, t.id) AS id,
                        COALESCE(1.0 / (60.0 + v.rank), 0.0) +
                        COALESCE(1.0 / (60.0 + t.rank), 0.0) AS rrf_score
                    FROM vector_ranked v
                    FULL OUTER JOIN text_ranked t ON v.id = t.id
                )
                SELECT
                    c.content,
                    d.source,
                    d.title,
                    rrf.rrf_score
                FROM rrf
                JOIN chunks c ON c.id = rrf.id
                JOIN documents d ON c.document_id = d.id
                ORDER BY rrf.rrf_score DESC
                LIMIT %s
            """, (embedding, embedding, question_text, top_k))
            return cur.fetchall()


def _search_vector_only(embed_text: str, top_k: int) -> list[tuple]:
    """Recherche vectorielle pure (chemin legacy, sans colonne tsv)."""
    embedding = get_embedding(embed_text)
    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.content, d.source, d.title,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """, (embedding, embedding, top_k))
            return cur.fetchall()


def search_chunks(question: str, top_k: int = TOP_K, hybrid: bool = True) -> list[tuple]:
    """
    Point d'entrée principal. Orchestre : HyDE → hybrid/vector → reranker.

    hybrid=True  : BM25 + vecteur + RRF (défaut, nécessite python db/init.py préalable)
    hybrid=False : recherche vectorielle pure (legacy S1)
    """
    if HYDE_ENABLED:
        from rag.hyde import hypothetical_document
        embed_text = hypothetical_document(question)
    else:
        embed_text = question

    candidate_k = RERANKER_TOP_N if RERANKER_BACKEND != "none" else top_k

    if hybrid:
        chunks = search_chunks_hybrid(embed_text, question, candidate_k)
    else:
        chunks = _search_vector_only(embed_text, candidate_k)

    if RERANKER_BACKEND != "none":
        from rag.reranker import rerank
        chunks = rerank(question, chunks, top_k=top_k)

    return chunks


def generate_answer(question: str, chunks: list) -> str:
    """Construit le prompt avec contexte et appelle Claude via OpenRouter."""
    context = "\n\n".join([
        f"[Source: {title or source}]\n{content}"
        for content, source, title, _ in chunks
    ])

    prompt = f"""Tu es un assistant expert. Réponds à la question UNIQUEMENT en utilisant le contexte fourni.
Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE:"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost",
        },
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    question = input("❓ Question : ")

    print("\n🔍 Recherche dans la base...")
    chunks = search_chunks(question)

    if not chunks:
        print("❌ Aucun chunk trouvé")
    else:
        print(f"✅ {len(chunks)} chunks trouvés")
        print("\n📝 Génération...")
        answer = generate_answer(question, chunks)

        print(f"\n{'='*60}\n{answer}\n{'='*60}")
        print("\n📚 Sources :")
        for i, (_, source, title, score) in enumerate(chunks, 1):
            print(f"   {i}. {title or source} (score: {score:.4f})")
