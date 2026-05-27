"""
scripts/reindex_contextual.py — Re-embed tous les chunks existants avec Contextual Retrieval.

Pour chaque document en base :
  1. Récupère tous ses chunks dans l'ordre (chunk_index ASC)
  2. Reconstruit le texte du document (join des contents)
  3. Pour chaque chunk, génère un contexte via Claude Haiku (avec prompt caching)
  4. Embedde f"{context}\\n\\n{chunk.content}"
  5. UPDATE chunks SET context=..., embedding=... WHERE id=...

Usage :
    python scripts/reindex_contextual.py                   # tous les documents
    python scripts/reindex_contextual.py --dry-run         # aperçu sans écriture
    python scripts/reindex_contextual.py --document-id 42  # un seul document
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import psycopg
from config import DB_CONNECTION_STRING, VOYAGE_MODEL
from utils import get_embedding
from rag.contextual import generate_chunk_context


def reindex_document(conn, document_id: int, title: str, dry_run: bool = False) -> int:
    """Re-indexe tous les chunks d'un document. Retourne le nombre de chunks traités."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, chunk_index, content FROM chunks WHERE document_id = %s ORDER BY chunk_index",
            (document_id,),
        )
        rows = cur.fetchall()

    if not rows:
        print(f"   ⚠️  Aucun chunk trouvé")
        return 0

    document_text = "\n\n".join(r[2] for r in rows)

    updates = []
    for chunk_id, chunk_index, content in rows:
        context = generate_chunk_context(document_text, content)
        embed_text = f"{context}\n\n{content}"
        embedding = get_embedding(embed_text)
        updates.append((context, embedding, VOYAGE_MODEL, chunk_id))
        print(f"   chunk {chunk_index:3d}: ✓ contexte généré")

    if not dry_run:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET context = %s, embedding = %s, embedding_model = %s WHERE id = %s",
                updates,
            )
        conn.commit()
        print(f"   💾 {len(updates)} chunks mis à jour en base")

    return len(updates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed chunks with contextual retrieval")
    parser.add_argument("--dry-run", action="store_true", help="Aperçu sans écriture en base")
    parser.add_argument("--document-id", type=int, default=None, help="Re-indexer un seul document")
    args = parser.parse_args()

    if args.dry_run:
        print("⚠️  Mode DRY-RUN — aucune écriture en base\n")

    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            if args.document_id:
                cur.execute("SELECT id, title FROM documents WHERE id = %s", (args.document_id,))
            else:
                cur.execute("SELECT id, title FROM documents ORDER BY id")
            docs = cur.fetchall()

        if not docs:
            print("❌ Aucun document en base.")
            return

        total = 0
        for doc_id, title in docs:
            print(f"\n📄 Document {doc_id}: {title or '(sans titre)'}")
            n = reindex_document(conn, doc_id, title or "", dry_run=args.dry_run)
            total += n

    suffix = " (dry-run)" if args.dry_run else ""
    print(f"\n✅ Total : {total} chunks traités{suffix}")


if __name__ == "__main__":
    main()
