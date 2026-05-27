"""
ingest.py — Pipeline d'ingestion d'un document dans le RAG.

Workflow :
  1. Télécharge le document
  2. UPSERT dans documents → récupère document_id
  3. Chunke le texte (SemanticChunker)
  4. Embed chaque chunk (Voyage)
  5. INSERT batch dans chunks
Tout dans UNE transaction : si ça crash en cours, rollback complet, pas de demi-état.
"""
import requests
import psycopg
from psycopg.types.json import Jsonb

from config import DB_CONNECTION_STRING, VOYAGE_MODEL, CONTEXTUAL_RETRIEVAL
from utils import get_embedding
from rag.chunker import chunk_text


def download_doc(url: str) -> str:
    print(f"📥 Téléchargement de {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def upsert_document(cursor, source: str, title: str | None = None, metadata: dict | None = None) -> int:
    """
    UPSERT dans documents et retourne l'id.
    
    ON CONFLICT (source) DO UPDATE : Postgres regarde la contrainte UNIQUE sur `source`.
      - Si source n'existe pas → INSERT classique.
      - Si source existe déjà → UPDATE le title/metadata + updated_at = NOW().
    
    RETURNING id : peu importe que ce soit un INSERT ou un UPDATE, on récupère
    l'id de la ligne (existante ou nouvelle).
    
    EXCLUDED est une pseudo-table qui contient les valeurs qu'on voulait INSERT
    (utile pour faire référence aux nouvelles valeurs dans la clause UPDATE).
    """
    metadata = metadata or {}
    
    cursor.execute("""
        INSERT INTO documents (source, title, metadata)
        VALUES (%s, %s, %s)
        ON CONFLICT (source) DO UPDATE
            SET title = EXCLUDED.title,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        RETURNING id
    """, (source, title, Jsonb(metadata)))
    
    document_id = cursor.fetchone()[0]
    
    # On supprime les anciens chunks pour repartir propre.
    # (Le CASCADE ne se déclenche que sur DELETE du document, pas sur UPDATE,
    # d'où ce DELETE explicite.)
    cursor.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    
    return document_id


def insert_chunks(cursor, document_id: int, chunks: list[str], document_text: str | None = None) -> None:
    """
    INSERT batch via executemany.

    Si CONTEXTUAL_RETRIEVAL=true et document_text fourni, Claude Haiku génère un contexte
    court pour chaque chunk (préfixé au chunk avant embedding, stocké dans chunks.context).
    """
    print(f"💾 Embedding + storage de {len(chunks)} chunks...")
    if CONTEXTUAL_RETRIEVAL and document_text:
        print("   🧠 Contextual Retrieval activé — génération des contextes via Claude Haiku...")
        from rag.contextual import generate_chunk_context

    rows = []
    for i, chunk in enumerate(chunks):
        context = None
        if CONTEXTUAL_RETRIEVAL and document_text:
            context = generate_chunk_context(document_text, chunk)
            embed_text = f"{context}\n\n{chunk}"
        else:
            embed_text = chunk

        embedding = get_embedding(embed_text)
        rows.append((document_id, i, chunk, context, embedding, VOYAGE_MODEL))
        if (i + 1) % 10 == 0:
            print(f"   → {i + 1}/{len(chunks)}")

    cursor.executemany("""
        INSERT INTO chunks (document_id, chunk_index, content, context, embedding, embedding_model)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, rows)


def ingest_text(text: str, source: str, title: str | None = None, metadata: dict | None = None) -> None:
    """Ingest un texte déjà disponible (lu depuis un fichier, par exemple)."""
    print(f"\n{'='*60}\nINGESTING: {source}\n{'='*60}")
    
    chunks = chunk_text(text)
    print(f"✂️  {len(chunks)} chunks créés (semantic)")
    
    if not chunks:
        print("⚠️  Aucun chunk généré, skip")
        return
    
    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            doc_id = upsert_document(cur, source, title, metadata)
            insert_chunks(cur, doc_id, chunks, document_text=text)

    print(f"✅ {source} ingéré (doc_id={doc_id})")


def ingest_url(url: str, source: str, title: str | None = None, metadata: dict | None = None) -> None:
    """Télécharge une URL puis l'ingest."""
    text = download_doc(url)
    ingest_text(text, source, title, metadata)


if __name__ == "__main__":
    docs = [
        {
            "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md",
            "source": "https://github.com/tiangolo/fastapi",
            "title": "FastAPI README",
            "metadata": {"type": "github_readme", "lang": "en", "topic": "web_framework"},
        },
        {
            "url": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md",
            "source": "https://github.com/langchain-ai/langgraph",
            "title": "LangGraph README",
            "metadata": {"type": "github_readme", "lang": "en", "topic": "agent_framework"},
        },
    ]
    
    for doc in docs:
        ingest_url(**doc)
    
    print("🎉 Tous les documents ingéré !")