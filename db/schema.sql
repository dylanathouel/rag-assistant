-- ============================================================
-- db/schema.sql
-- Idempotent : peut être ré-exécuté sans casser quoi que ce soit
-- ============================================================

-- pgvector fournit le type `vector` et les opérateurs <=> (cosine), <-> (L2), <#> (inner product)
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- TABLE: documents
-- 1 ligne = 1 source ingérée (PDF, URL, fichier .md...)
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,

    -- source = identifiant naturel (URL ou chemin). UNIQUE = on ne ré-ingère pas deux fois.
    -- Si tu veux ré-indexer, tu DELETE FROM documents WHERE source = ? puis INSERT.
    -- Le CASCADE sur chunks (plus bas) supprime automatiquement les chunks associés.
    source       TEXT NOT NULL UNIQUE,

    title        TEXT,

    -- metadata = tout ce qui est variable selon la source.
    -- Exemples : {"author": "...", "lang": "fr", "type": "pdf", "pages": 42, "tags": ["legal"]}
    -- jsonb (pas json) car indexable et plus rapide à parser.
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- TABLE: chunks
-- 1 ligne = 1 morceau d'un document, prêt à être recherché
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id               SERIAL PRIMARY KEY,

    -- FK vers documents.id
    -- ON DELETE CASCADE : si je supprime le doc parent, ses chunks disparaissent aussi
    document_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Position du chunk dans le document (0, 1, 2, ...).
    -- Sert à reconstruire l'ordre et à récupérer les chunks voisins (S2: Contextual Retrieval).
    chunk_index      INTEGER NOT NULL,

    -- Le texte du chunk lui-même
    content          TEXT NOT NULL,

    -- S2 : Contextual Retrieval (Anthropic). Pour chaque chunk, Claude Haiku génère un court contexte
    -- qui est PRÉFIXÉ au content avant l'embedding. On garde le contexte ici pour audit + ré-embedding.
    -- Nullable car non rempli en S1.
    context          TEXT,

    -- Le vecteur. 1024 = dim de voyage-3.
    -- Si tu changes de modèle d'embedding plus tard, la dim peut changer (ex: voyage-3-large = 1024,
    -- openai text-embedding-3-large = 3072). D'où la colonne embedding_model en dessous.
    embedding        VECTOR(1024),

    -- Tracer quel modèle a généré ce vecteur. Crucial le jour où tu migres :
    -- tu peux ré-embedder seulement les lignes "obsolètes" sans tout refaire.
    embedding_model  TEXT NOT NULL DEFAULT 'voyage-3',

    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Garde-fou : pas deux fois le chunk #5 du même document.
    -- Crée AUSSI un index B-tree implicite sur (document_id, chunk_index)
    -- → accélère "SELECT ... WHERE document_id = X AND chunk_index BETWEEN a AND b ORDER BY chunk_index"
    -- qu'on utilisera en S2 pour récupérer les chunks voisins.
    UNIQUE (document_id, chunk_index)
);


-- ============================================================
-- INDEX: recherche vectorielle (HNSW > IVFFlat par défaut)
-- ============================================================
-- vector_cosine_ops car on utilise la cosine similarity (opérateur <=>).
-- Si tu utilisais L2 distance, ce serait vector_l2_ops. Si inner product, vector_ip_ops.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops);


-- ============================================================
-- INDEX: filtres sur metadata (jsonb)
-- ============================================================
-- GIN = Generalized Inverted Index. Idéal pour jsonb avec opérateurs @>, ?, ?&, ?|.
-- Exemple S2: WHERE chunks.metadata @> '{"section": "introduction"}'
-- Sans cet index = full scan. Avec = quasi-instantané.
CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx
    ON chunks
    USING gin (metadata);

CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
    ON documents
    USING gin (metadata);


-- ============================================================
-- TRIGGER: auto-update documents.updated_at à chaque UPDATE
-- ============================================================
-- Postgres ne le fait pas tout seul. Pattern standard avec une fonction + un trigger.
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- DROP IF EXISTS car CREATE TRIGGER n'a pas de IF NOT EXISTS (Postgres < 17)
DROP TRIGGER IF EXISTS documents_set_updated_at ON documents;
CREATE TRIGGER documents_set_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();