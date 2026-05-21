"""
query.py — Recherche les chunks pertinents et génère une réponse.
"""
import requests
import psycopg

from config import DB_CONNECTION_STRING, OPENROUTER_API_KEY, LLM_MODEL, TOP_K
from utils import get_embedding


def search_chunks(question: str, top_k: int = TOP_K):
    """
    Recherche les top_k chunks les plus proches.
    
    Note importante : on `ORDER BY c.embedding <=> %s::vector` directement
    (et non `ORDER BY similarity DESC`). Pourquoi ? Parce que l'index HNSW
    accélère l'opérateur <=> directement. Si on ORDER BY sur une expression
    calculée (1 - similarity), le planner Postgres peut ne pas utiliser l'index.
    """
    embedding = get_embedding(question)
    
    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    c.content,
                    d.source,
                    d.title,
                    1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """, (embedding, embedding, top_k))
            
            return cur.fetchall()


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
        for i, (_, source, title, sim) in enumerate(chunks, 1):
            print(f"   {i}. {title or source} (sim: {sim:.3f})")