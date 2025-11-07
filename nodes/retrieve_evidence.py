# nodes/retrieve_evidence.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path

# Кэш для FAISS
FAISS_CACHE_DIR = Path("cache/faiss")
FAISS_CACHE_DIR.mkdir(exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def retrieve_evidence(state):
    """
    Узел 4: Для каждой гипотезы находит релевантные фрагменты текста через векторный поиск.
    Использует FAISS с кэшированием.
    """
    print("🔎 Узел: Поиск доказательств (FAISS + кэш)...")
    
    hypotheses = state.get("hypotheses", [])
    texts = state.get("retrieved_texts", [])
    
    if not hypotheses or not texts:
        print("⚠️ Нет гипотез или текстов для поиска.")
        return {"evidence": []}
    
    # Разбиваем тексты на чанки
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = []
    for text in texts:
        if text.strip():
            chunks.extend(splitter.split_text(text))
    
    if not chunks:
        print("⚠️ Нет валидных чанков после разделения.")
        return {"evidence": []}

    # 🔥 Кэшируем FAISS по хешу текстов
    import hashlib
    cache_key = hashlib.md5("".join(chunks).encode()).hexdigest()
    cache_path = FAISS_CACHE_DIR / cache_key

    if (cache_path / "index.faiss").exists():
        print("💾 Загружаем кэшированный FAISS индекс...")
        vectorstore = FAISS.load_local(
            str(cache_path),
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        print("🏗️ Создаём новый FAISS индекс...")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding_model)
        print("💾 Сохраняем FAISS индекс в кэш...")
        vectorstore.save_local(cache_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    evidence = []
    for hypothesis in hypotheses:
        print(f"🔍 Поиск по гипотезе: '{hypothesis[:60]}...'")
        docs = retriever.invoke(hypothesis)
        found_chunks = [{"text": doc.page_content} for doc in docs]
        
        evidence.append({
            "hypothesis": hypothesis,
            "chunks": found_chunks
        })

    print(f"✅ Найдены доказательства для {len(evidence)} гипотез")
    return {"evidence": evidence}