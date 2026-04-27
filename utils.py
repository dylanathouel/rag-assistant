import requests
from config import VOYAGE_API_KEY, VOYAGE_MODEL

def get_embedding(text):
    """Crée un embedding via Voyage."""
    response = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
        json={"input": [text], "model": VOYAGE_MODEL}
    )
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        raise Exception(f"Erreur Voyage: {response.text}")