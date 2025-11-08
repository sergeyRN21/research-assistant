# nodes/retrieve_evidence.py
from langchain_community.vectorstores import FAISS
# nodes/retrieve_evidence.py (с использованием embedding_loader)
from .embedding_loader import get_embedding_model # Импорт из соседнего файла

# embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5") # Удалить эту строку
embedding_model = get_embedding_model() # Загрузить через функцию

# ... остальной код остаётся тем же

def retrieve_evidence(state):
    """
    Узел: Для каждой гипотезы из `state["hypotheses"]` находит релевантные чанки.
    Использует multi-query из `state["queries"]` для поиска по каждому гипотезному запросу.
    """
    print("🔎 Узел: Поиск доказательств по гипотезам...")

    hypotheses = state.get("hypotheses", [])
    queries = state.get("queries", []) # Multi-query варианты
    chunks_data = state.get("chunks_with_metadata", [])

    if not hypotheses or not chunks_data:
        print("⚠️ Нет гипотез или чанков для поиска доказательств.")
        return {"evidence": []}

    # Подготовка текстов и метаданных
    texts = [chunk["text"] for chunk in chunks_data]
    metadatas = [chunk.get("metadata", {}) for chunk in chunks_data]

    # Создаём векторное хранилище
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Функция: объединение без дубликатов
    def get_unique_union(docs_lists):
        seen = set()
        unique_docs = []
        for docs in docs_lists:
            for doc in docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen:
                    seen.add(content_hash)
                    unique_docs.append(doc)
        return unique_docs

    evidence_list = []

    # Для каждой гипотезы
    for hypothesis in hypotheses:
        print(f"🔍 Поиск доказательств для гипотезы: '{hypothesis}'")
        # Собираем все поисковые запросы (оригинальная гипотеза + multi-query варианты)
        search_queries = [hypothesis] + queries

        all_docs_for_hyp = []
        for query in search_queries:
            print(f"   📄 Поиск по запросу: '{query}'")
            docs = retriever.invoke(query)
            all_docs_for_hyp.append(docs)

        # Объединяем результаты для этой гипотезы без дубликатов
        unique_docs_for_hyp = get_unique_union(all_docs_for_hyp)

        # Формируем список чанков для этой гипотезы
        chunks_for_hyp = [{"text": doc.page_content, "metadata": doc.metadata} for doc in unique_docs_for_hyp]

        # Добавляем в итоговый список evidence
        evidence_list.append({
            "hypothesis": hypothesis,
            "chunks": chunks_for_hyp
        })

    print(f"✅ Сформировано {len(evidence_list)} элементов доказательства.")
    return {"evidence": evidence_list}