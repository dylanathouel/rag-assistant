import requests
import psycopg
from utils import get_embedding
from config import (
    DB_CONNECTION_STRING,
    OPENROUTER_API_KEY,
    LLM_MODEL,
    TOP_K
)


def search_chunks(question, top_k=TOP_K):
    """Recherche les chunks les plus proches de la question."""
    embedding = get_embedding(question)
    
    conn = psycopg.connect(DB_CONNECTION_STRING)
    cursor = conn.cursor()
    
    # Recherche les top_k chunks avec cosine similarity
    cursor.execute(f"""
    SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity
    FROM chunks
    ORDER BY similarity DESC
    LIMIT {top_k}
    """, (embedding,))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results

def generate_answer(question, chunks):
    """Appelle Claude pour générer la réponse."""
    context = "\n\n".join([f"[Source: {chunk[1]}]\n{chunk[0]}" for chunk in chunks])
    
    prompt = f"""Tu es un assistant expert en FastAPI et LangGraph.
Réponds à la question UNIQUEMENT en utilisant le contexte fourni ci-dessous.
Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE:"""
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost"
        },
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Erreur OpenRouter: {response.text}")

# ===== MAIN =====
if __name__ == "__main__":
    question = input("❓ Question : ")
    
    print("\n🔍 Recherche dans la base...")
    chunks = search_chunks(question, top_k=TOP_K)
    
    if not chunks:
        print("❌ Aucun chunk trouvé")
    else:
        print(f"✅ {len(chunks)} chunks trouvés")
        print("\n📝 Génération de la réponse...")
        answer = generate_answer(question, chunks)
        
        print("\n" + "="*60)
        print(answer)
        print("="*60)
        
        print("\n📚 Sources :")
        for i, (content, source, similarity) in enumerate(chunks, 1):
            print(f"   {i}. {source} (similarity: {similarity:.2f})")