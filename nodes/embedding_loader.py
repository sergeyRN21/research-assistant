# nodes/embedding_loader.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Lazy-load эмбеддинги — только при вызове.
    """
    print("📦 Загружаем эмбеддинги BAAI/bge-large-en-v1.5...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    print("✅ Эмбеддинги загружены.")
    return embedding_model