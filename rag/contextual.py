"""
rag/contextual.py — Génère un contexte court pour chaque chunk via Claude Haiku
avec prompt caching (Anthropic SDK direct).

Prompt caching :
  - document_text est passé en premier bloc avec cache_control="ephemeral"
  - Tous les chunks d'un même document partagent ce cache après le 1er appel
  - Coût réel ≈ 1 plein appel par document + N appels bon marché pour les chunks suivants
"""
import anthropic
from config import ANTHROPIC_API_KEY, CONTEXT_MODEL


def generate_chunk_context(document_text: str, chunk_text: str) -> str:
    """
    Retourne 1-2 phrases situant le chunk dans son document source.
    Le document_text est mis en cache par Anthropic (cache_control ephemeral, TTL 5 min).
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=CONTEXT_MODEL,
        max_tokens=150,
        system=(
            "Generate a brief 1-2 sentence context that situates the following chunk "
            "within its source document. Be concise and factual. "
            "Output only the context sentence(s), no preamble."
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{document_text}\n</document>\n\nChunk to situate:",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"<chunk>\n{chunk_text}\n</chunk>",
                    },
                ],
            }
        ],
    )
    return response.content[0].text.strip()
