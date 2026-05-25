# CLAUDE.md

## Stack

| Composant | Technologie | Version |
|---|---|---|
| Langage | Python | 3.12+ |
| API REST | FastAPI + Uvicorn | 0.115.4 / 0.32.0 |
| UI | Streamlit | 1.39.0 |
| Base de données | PostgreSQL 17 + pgvector | — |
| Embeddings | Voyage AI (`voyage-3`, dim 1024) | langchain-voyageai 0.1.3 |
| LLM | Claude via OpenRouter (`anthropic/claude-sonnet-4.6`) | — |
| Chunking | LangChain SemanticChunker (percentile 95) | langchain-experimental 0.3.3 |
| Driver DB | psycopg v3 (binary) | 3.2.1 |

Pas de tests en S1 — ne pas en créer sauf demande explicite.

---

## Démarrage

```bash
# 1. Lancer Postgres (Docker)
docker compose up -d postgres

# 2. Appliquer le schéma (idempotent — safe à ré-exécuter)
python db/init.py

# 3. API FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. UI Streamlit (autre terminal)
streamlit run streamlit_app.py
```

**Stack complète via Docker Compose :**
```bash
docker compose up        # API sur :8000 + Postgres sur :5432
# Streamlit doit tourner séparément (pas dans le compose)
```

---

## Variables d'environnement (`.env`)

```env
VOYAGE_API_KEY=...
OPENROUTER_API_KEY=...
LLM_MODEL=anthropic/claude-sonnet-4.6   # optionnel, c'est le défaut

DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=...
DB_PASSWORD=...
```

- `config.py` charge tout via `python-dotenv`. Valeurs par défaut DB : `localhost:5432/rag_db`.
- En Docker Compose, `DB_HOST=postgres` (nom du service).
- `streamlit_app.py` lit `API_URL` (défaut `http://localhost:8000`).

---

## Ingestion

```bash
# Ingest deux URLs de démo (FastAPI + LangGraph READMEs)
python ingest.py

# Bulk ingest d'un dossier .md
python scripts/ingest_corpus.py corpus/fastapi --topic web_framework --skip-existing

# Options du bulk ingest :
#   --pattern       glob pattern (défaut: **/*.md)
#   --exclude       pattern à exclure (répétable)
#   --topic         valeur dans metadata.topic (défaut: general)
#   --max-bytes     skip les fichiers > N bytes (défaut: 200 000 ≈ 50k tokens)
#   --skip-existing skip les sources déjà en DB

# Via l'API
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://...", "source": "unique-id", "title": "Mon doc", "metadata": {}}'
```

---

## Query

```bash
# CLI interactif
python query.py

# Via l'API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?", "top_k": 5}'
```

---

## API — Endpoints

| Méthode | Chemin | Description |
|---|---|---|
| GET | `/` | Info app + lien docs |
| GET | `/health` | État Postgres : nb docs + nb chunks |
| POST | `/query` | Recherche RAG → réponse Claude |
| POST | `/ingest/url` | Ingère une URL |

Doc OpenAPI interactive : `http://localhost:8000/docs`

**`POST /query`**
```json
// Request  { question: string, top_k: int 1-20 (défaut 5) }
// Response
{
  "question": "...",
  "answer": "...",
  "sources": [
    { "source": "...", "title": "...", "similarity": 0.87, "content_preview": "..." }
  ]
}
```

**`POST /ingest/url`**
```json
// Request  { url, source (identifiant unique), title?, metadata? }
// Response { source, document_id, chunks_count }
```

---

## Architecture

### Pipeline d'ingestion

```
Document (URL ou fichier texte)
    → chunk_text()          # SemanticChunker : coupe sur frontières sémantiques
    → get_embedding()       # Voyage API : vecteur 1024 dims par chunk
    → UPSERT documents      # source UNIQUE → idempotent (UPDATE si existe déjà)
    → DELETE old chunks     # nettoyage explicite avant ré-ingestion
    → INSERT chunks batch   # executemany (1 seul round-trip réseau)
```

Tout dans **une seule transaction psycopg** : rollback complet si crash en cours.

### Pipeline de query

```
Question
    → get_embedding()                   # même modèle que l'ingestion
    → ORDER BY embedding <=> %s::vector # HNSW cosine (index utilisé directement)
    → generate_answer()                 # prompt + contexte → OpenRouter → Claude
```

### SemanticChunker (`rag/chunker.py`)

Paramètres : `breakpoint_threshold_type="percentile"`, `breakpoint_threshold_amount=95` — on coupe sur les 5 % plus gros sauts sémantiques entre phrases consécutives (embedding de chaque phrase via Voyage).

Les instances `_embeddings` et `_chunker` sont créées **une seule fois au chargement du module** — ne pas les déplacer dans la fonction `chunk_text`.

---

## Schéma DB

### Table `documents` — 1 ligne par source ingérée

```
id          SERIAL PRIMARY KEY
source      TEXT NOT NULL UNIQUE    -- identifiant naturel (URL, chemin fichier)
title       TEXT
metadata    JSONB NOT NULL DEFAULT '{}'
created_at  TIMESTAMPTZ
updated_at  TIMESTAMPTZ             -- auto-mis à jour par trigger set_updated_at()
```

### Table `chunks` — 1 fragment par ligne

```
id               SERIAL PRIMARY KEY
document_id      INTEGER REFERENCES documents(id) ON DELETE CASCADE
chunk_index      INTEGER NOT NULL           -- position dans le doc (0-based)
content          TEXT NOT NULL
context          TEXT                       -- S2 : Contextual Retrieval (NULL en S1)
embedding        VECTOR(1024)
embedding_model  TEXT NOT NULL DEFAULT 'voyage-3'
metadata         JSONB NOT NULL DEFAULT '{}'
created_at       TIMESTAMPTZ
UNIQUE (document_id, chunk_index)           -- crée aussi un index B-tree implicite
```

### Index

| Nom | Table | Type | Utilisation |
|---|---|---|---|
| `chunks_embedding_hnsw_idx` | chunks | HNSW cosine | recherche ANN via `<=>` |
| `chunks_metadata_gin_idx` | chunks | GIN | filtres `@>` / `?` sur metadata |
| `documents_metadata_gin_idx` | documents | GIN | filtres `@>` / `?` sur metadata |

---

## ⚠️ Contraintes critiques — ne pas modifier

**`ORDER BY c.embedding <=> %s::vector`** dans `query.py` :  
Ne **pas** réécrire en `ORDER BY similarity DESC` ou sur une expression calculée (`1 - ...`). L'index HNSW accélère l'opérateur `<=>` directement ; toute réécriture sur une expression dérivée empêche l'optimiseur Postgres d'utiliser l'index.

**`DELETE FROM chunks WHERE document_id = %s`** dans `ingest.py` :  
Le `ON DELETE CASCADE` de la FK ne se déclenche que sur `DELETE` du document parent, pas sur `UPDATE`. L'UPSERT mettant à jour le document, le delete des anciens chunks doit rester explicite.

---

## Structure des fichiers

```
.
├── config.py                  # Config centrale : clés API, params RAG, DB_CONNECTION_STRING
├── utils.py                   # get_embedding() — appel direct API Voyage
├── ingest.py                  # Pipeline ingestion + CLI (démo 2 URLs)
│                              #   ingest_url(), ingest_text(), upsert_document(), insert_chunks()
├── query.py                   # search_chunks() + generate_answer() + CLI interactif
├── streamlit_app.py           # UI : appelle l'API via HTTP, sidebar health + slider top_k
│
├── rag/
│   └── chunker.py             # chunk_text() — SemanticChunker, singletons module-level
│
├── api/
│   ├── main.py                # App FastAPI, 3 routers montés avec préfixes
│   ├── schemas.py             # Modèles Pydantic (QueryRequest/Response, IngestUrlRequest/Response)
│   └── routers/
│       ├── health.py          # GET /health — compte docs + chunks
│       ├── query.py           # POST /query — search + generate
│       └── ingest.py          # POST /ingest/url — délègue à ingest.ingest_url()
│
├── db/
│   ├── schema.sql             # Schéma idempotent : tables, HNSW, GIN, trigger updated_at
│   └── init.py                # Applique schema.sql à la DB
│
├── scripts/
│   └── ingest_corpus.py       # Bulk ingest local (.md) avec filtres argparse
│
├── corpus/
│   └── fastapi/               # Corpus de démo : docs FastAPI
│
├── Dockerfile                 # python:3.12-slim, CMD bash
├── docker-compose.yml         # rag-app (:8000) + postgres pgvector/pg17 (:5432)
└── requirements.txt           # Dépendances épinglées
```

---

## Roadmap S2 (colonnes déjà présentes dans le schéma)

- **Contextual Retrieval** : colonne `context` dans `chunks` — Claude Haiku génère un court contexte préfixé au chunk avant embedding, stocké ici pour audit et ré-embedding.
- **Chunks voisins** : `chunk_index ± N` via l'index B-tree implicite sur `(document_id, chunk_index)`.
- **Filtres metadata** : opérateurs `@>` / `?` sur `chunks.metadata` / `documents.metadata` (index GIN en place).
- **Re-embedding sélectif** : colonne `embedding_model` — cibler uniquement les chunks avec l'ancien modèle lors d'une migration.
- **Évaluation RAGAS** : tuning du `breakpoint_threshold_amount` du SemanticChunker.
