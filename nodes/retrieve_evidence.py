# nodes/retrieve_evidence.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def retrieve_evidence(state):
    """
    Узел 4: Для каждого запроса находит релевантные чанки через FAISS.
    Использует метаданные для фильтрации (например, только где есть 'results').
    """
    print("🔎 Узел: Поиск доказательств с фильтрацией по метаданным...")

    queries = state.get("queries", [])
    chunks_data = state.get("chunks_with_metadata", [])

    if not queries or not chunks_data:
        print("⚠️ Нет запросов или чанков.")
        return {"evidence": []}

    # Подготовка данных для FAISS
    texts = [chunk["text"] for chunk in chunks_data]
    metadatas = [chunk["metadata"] for chunk in chunks_data]

    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    
    # Функция: уникальное объединение документов
    def get_unique_union(docs_list):
        seen = set()
        unique_docs = []
        for doc in docs_list:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        return unique_docs

    evidence_items = []
    for query in queries:
        print(f"🔍 Поиск по запросу: '{query[:60]}...'")

        # 🔥 Ищем только в чанках с результатами или методами
        docs = vectorstore.similarity_search_with_score(
            query,
            k=3,
            filter=lambda m: m.get("contains_results", False) or m.get("contains_method", False)
        )
        
        # Убираем дубликаты
        unique_docs = get_unique_union([doc for doc, _ in docs])

        found_chunks = []
        for doc in unique_docs:
            found_chunks.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        evidence_items.append({
            "hypothesis": f"Relevant fragment (query-translated): {query}",  # можно заменить на генерацию гипотез
            "chunks": found_chunks
        })

    print(f"✅ Найдены доказательства для {len(evidence_items)} запросов")
    return {"evidence": evidence_items}