# nodes/retrieve_evidence.py
from langchain_community.vectorstores import FAISS
from nodes.embedding_loader import get_embedding_model

def retrieve_evidence(state):
    print("🔎 Узел: Поиск доказательств по нескольким запросам...")

    queries = state.get("queries", [])
    chunks_data = state.get("chunks_with_metadata", [])

    if not queries or not chunks_data:
        print("⚠️ Нет запросов или чанков.")
        return {"evidence": []}

    # 🔥 Lazy load эмбеддингов — только при вызове
    embedding_model = get_embedding_model()

    # Подготовка данных для FAISS
    texts = [chunk["text"] for chunk in chunks_data]
    metadatas = [chunk.get("metadata", {}) for chunk in chunks_data]

    # Создаём векторное хранилище
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # Функция: уникальное объединение документов
    def get_unique_union(doc_lists):
        seen = set()
        unique_docs = []
        for docs in doc_lists:
            for doc in docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen:
                    seen.add(content_hash)
                    unique_docs.append(doc)
        return unique_docs

    # Выполняем поиск по каждому запросу
    all_docs = []
    for query in queries:
        print(f"🔍 Поиск: '{query}'")
        docs = retriever.invoke(query)
        all_docs.append(docs)

    # Объединяем без дубликатов
    unique_docs = get_unique_union(all_docs)

    # Формируем evidence
    evidence_items = []
    for i, doc in enumerate(unique_docs):
        evidence_items.append({
            "hypothesis": f"Relevant fragment (query-translated) {i+1}",
            "chunks": [{
                "text": doc.page_content,
                "metadata": getattr(doc, "metadata", {})
            }]
        })

    print(f"✅ Найдено {len(unique_docs)} уникальных фрагментов")
    return {"evidence": evidence_items}