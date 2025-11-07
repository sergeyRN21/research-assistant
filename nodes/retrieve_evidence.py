# nodes/retrieve_evidence.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def retrieve_evidence(state):
    """
    Узел 4: Для каждой гипотезы находит релевантные чанки через векторный поиск.
    Использует метаданные для фильтрации (например, только где есть "results").
    """
    print("🔎 Узел: Поиск доказательств (с фильтрацией по метаданным)...")
    
    hypotheses = state.get("hypotheses", [])
    chunks_data = state.get("chunks_with_metadata", [])
    
    if not hypotheses or not chunks_data:
        print("⚠️ Нет гипотез или чанков для поиска.")
        return {"evidence": []}

    # Создаём векторное хранилище
    texts = [chunk["text"] for chunk in chunks_data]
    metadatas = [chunk["metadata"] for chunk in chunks_data]

    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    
    evidence = []
    for hypothesis in hypotheses:
        print(f"🔍 Поиск по гипотезе: '{hypothesis[:60]}...'")

        # 🔥 Фильтруем: ищем только в чанках с "results" или "experiment"
        docs = vectorstore.similarity_search(
            hypothesis,
            k=3,
            filter=lambda m: m.get("contains_results", False) or m.get("contains_experiment", False)
        )
        
        found_chunks = []
        for doc in docs:
            found_chunks.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        evidence.append({
            "hypothesis": hypothesis,
            "chunks": found_chunks
        })

    print(f"✅ Найдены доказательства для {len(evidence)} гипотез")
    return {"evidence": evidence}