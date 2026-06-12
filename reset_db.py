import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

url = os.getenv("QDRANT_URL") or "http://localhost:6333"
api_key = os.getenv("QDRANT_KEY") or None
if not api_key or api_key.strip() == "":
    api_key = None

print(f"🔌 Conectando a Qdrant en {url}...")
client = QdrantClient(url=url, api_key=api_key)

collections = ["histo_chunks", "histo_imagenes", "memoria_histo"]
for col in collections:
    try:
        if client.collection_exists(col):
            client.delete_collection(col)
            print(f"🗑️ Colección eliminada con éxito: {col}")
        else:
            print(f"ℹ️ La colección '{col}' no existe. Omitiendo.")
    except Exception as e:
        print(f"❌ Error al eliminar la colección {col}: {e}")

print("✨ Reinicio de base de datos Qdrant completado.")
