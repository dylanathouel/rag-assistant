"""
scripts/ingest_corpus.py — Bulk ingest des .md d'un dossier, avec filtres.
"""
import argparse
import fnmatch
from pathlib import Path

import psycopg
from config import DB_CONNECTION_STRING
from ingest import ingest_text


def existing_sources() -> set[str]:
    """Récupère les sources déjà en DB pour pouvoir les skipper."""
    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source FROM documents")
            return {row[0] for row in cur.fetchall()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--pattern", default="**/*.md")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Pattern à exclure (peut être répété)")
    parser.add_argument("--topic", default="general")
    parser.add_argument("--max-bytes", type=int, default=200_000,
                        help="Skip les fichiers > N bytes (défaut 200k = ~50k tokens)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip les sources déjà en DB")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.exists():
        print(f"❌ Dossier introuvable : {root}")
        return

    files = sorted(root.glob(args.pattern))

    # Filtres
    if args.exclude:
        files = [
            f for f in files
            if not any(fnmatch.fnmatch(f.name, pat) for pat in args.exclude)
        ]
    files_too_big = [f for f in files if f.stat().st_size > args.max_bytes]
    files = [f for f in files if f.stat().st_size <= args.max_bytes]

    existing = existing_sources() if args.skip_existing else set()

    print(f"📚 {len(files)} fichiers retenus")
    if files_too_big:
        print(f"   ({len(files_too_big)} skippés car > {args.max_bytes} bytes : "
              f"{', '.join(f.name for f in files_too_big[:5])}{'...' if len(files_too_big) > 5 else ''})")
    print()

    ok, skipped, errors = 0, 0, 0
    for i, file in enumerate(files, 1):
        source = str(file.relative_to(root.parent))

        if source in existing:
            print(f"[{i}/{len(files)}] ⏭  {file.name} déjà en DB, skip")
            skipped += 1
            continue

        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                print(f"[{i}/{len(files)}] ⏭  {file.name} vide, skip")
                skipped += 1
                continue

            title = file.stem
            metadata = {
                "topic": args.topic,
                "filename": file.name,
                "size_bytes": file.stat().st_size,
            }

            print(f"[{i}/{len(files)}]", end=" ")
            ingest_text(text, source=source, title=title, metadata=metadata)
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(files)}] ❌ {file.name} : {e}")
            errors += 1

    print(f"\n🎉 Bilan : {ok} ingérés, {skipped} skipped, {errors} erreurs")


if __name__ == "__main__":
    main()