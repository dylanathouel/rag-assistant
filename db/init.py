"""
db/init.py — Applique le schéma SQL à la DB.
Idempotent : peut être ré-exécuté sans rien casser (les CREATE TABLE IF NOT EXISTS protègent).
"""
from pathlib import Path
import psycopg
from config import DB_CONNECTION_STRING

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db():
    schema_sql = SCHEMA_PATH.read_text()
    print(f"📄 Lecture du schéma depuis {SCHEMA_PATH}")

    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    print("✅ Schéma appliqué")


if __name__ == "__main__":
    init_db()