"""
rag/hyde.py — HyDE : Hypothetical Document Embeddings.

Au lieu d'embedder la question brute, on demande au LLM de générer une réponse
hypothétique (3-5 phrases style documentation), puis on embedde ce paragraphe.
Ce texte est beaucoup plus proche dans l'espace vectoriel des vrais chunks de doc.

Note : le BM25 (côté text_ranked) utilise toujours la question originale — les
mots-clés doivent matcher les termes de l'utilisateur, pas la paraphrase du LLM.
"""
import requests
from config import OPENROUTER_API_KEY, LLM_MODEL


def hypothetical_document(question: str) -> str:
    """
    Génère un paragraphe de réponse hypothétique à embedder à la place de la question.
    Utilise le LLM existant via OpenRouter — aucune clé API supplémentaire.
    """
    prompt = (
        "Write a short paragraph (3-5 sentences) that would be the ideal answer to the "
        "following question, written as if extracted from technical documentation. "
        "Always generate a plausible answer even if uncertain — never say you don't know.\n\n"
        f"Question: {question}\n\n"
        "Answer paragraph:"
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
