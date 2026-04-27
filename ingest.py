import requests
import psycopg
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    VOYAGE_API_KEY, 
    VOYAGE_MODEL, 
    DB_CONNECTION_STRING,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from utils import get_embedding

def create_table():
    """Crée la table chunks si elle n'existe pas."""
    conn = psycopg.connect(DB_CONNECTION_STRING)
    cursor = conn.cursor()
    
    print("📦 Création de la table chunks...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            embedding vector(1024)
        );
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
        ON chunks 
        USING hnsw (embedding vector_cosine_ops);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Table prête !\n")

def download_doc(url):
    """Télécharge un document."""
    print(f"📥 Téléchargement de {url}...")
    response = requests.get(url)
    return response.text

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Découpe le texte en chunks sémantiques."""
    print(f"✂️  Chunking ({chunk_size} chars, overlap {chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    print(f"   → {len(chunks)} chunks créés")
    return chunks

def store_chunks(chunks, source):
    """Stocke les chunks avec leurs embeddings dans Postgres."""
    conn = psycopg.connect(DB_CONNECTION_STRING)
    cursor = conn.cursor()
    
    # 🗑️ SUPPRIMER tous les chunks avec cette source
    print(f"🗑️  Suppression des chunks existants pour {source}...")
    cursor.execute("DELETE FROM chunks WHERE source = %s", (source,))
    
    # 💾 Ajouter les nouveaux
    print(f"💾 Stockage des embeddings...")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        cursor.execute(
            "INSERT INTO chunks (content, source, embedding) VALUES (%s, %s, %s)",
            (chunk, source, embedding)
        )
        if (i + 1) % 10 == 0:
            print(f"   → {i + 1}/{len(chunks)}")
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ {len(chunks)} chunks stockés pour {source}!\n")

# ===== MAIN =====
if __name__ == "__main__":
    # 1️⃣ CRÉER LA TABLE D'ABORD
    create_table()
    
    # 2️⃣ INGEST les docs
    docs = [
        {
            "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md",
            "source": "FastAPI README"
        },
        {
            "url": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md",
            "source": "LangGraph README"
        }
    ]
    
    for doc in docs:
        print(f"\n{'='*60}")
        print(f"INGESTING: {doc['source']}")
        print(f"{'='*60}")
        
        text = download_doc(doc['url'])
        chunks = chunk_text(text)
        store_chunks(chunks, doc['source'])
    
    print("\n🎉 Tous les documents ingested !")