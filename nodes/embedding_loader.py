# nodes/embedding_loader.py
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Lazy-load эмбеддинги — только при вызове.
    Важно: не используем GPU, чтобы избежать ошибки в Streamlit Cloud.
    """
    print("📦 Загружаем эмбеддинги BAAI/bge-large-en-v1.5 (CPU-only)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},  # 🔥 Важно: не cuda!
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Эмбеддинги загружены.")
    return embedding_model