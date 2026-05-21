"""
rag/chunker.py — Semantic chunking via langchain_experimental.

Remplace RecursiveCharacterTextSplitter pour avoir des chunks coupés
sur des frontières sémantiques (changements de sujet) plutôt qu'à
longueur fixe.
"""
from langchain_voyageai import VoyageAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from config import VOYAGE_API_KEY, VOYAGE_MODEL


# === Instances créées UNE SEULE FOIS au chargement du module ===
# (Voir explication en bas du fichier sur le "pourquoi module-level")

_embeddings = VoyageAIEmbeddings(
    model=VOYAGE_MODEL,
    voyage_api_key=VOYAGE_API_KEY,
)

_chunker = SemanticChunker(
    embeddings=_embeddings,
    # "percentile" + 95 : on coupe sur les 5% plus gros sauts sémantiques.
    # C'est le défaut recommandé par LangChain, à tuner plus tard avec RAGAS en S2.
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)


def chunk_text(text: str) -> list[str]:
    """
    Découpe un texte en chunks sémantiques.

    Sous le capot, le SemanticChunker :
      1. split le texte en phrases
      2. embed chaque phrase via Voyage (N appels API — c'est le coût)
      3. calcule la distance cosine entre phrases consécutives
      4. coupe quand la distance dépasse le percentile 95 (= saut de sujet)

    Retourne une liste de strings, donc 100% compatible avec
    ton store_chunks(chunks, source) existant.
    """
    if not text.strip():
        return []

    docs = _chunker.create_documents([text])
    return [d.page_content for d in docs if d.page_content.strip()]