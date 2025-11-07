# nodes/retrieve_evidence.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Настройка эмбеддингов
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def retrieve_evidence(state):
    """
    Узел 4: Для каждой гипотезы находит релевантные фрагменты текста через векторный поиск.
    Использует FAISS вместо Chroma — для совместимости со Streamlit Cloud.
    
    Вход: state["hypotheses"], state["retrieved_texts"]
    Выход: state["evidence"] = [{"hypothesis": "...", "chunks": [...]}]
    """
    print("🔎 Узел: Поиск доказательств (FAISS)...")
    
    hypotheses = state.get("hypotheses", [])
    texts = state.get("retrieved_texts", [])
    
    if not hypotheses or not texts:
        print("⚠️ Нет гипотез или текстов для поиска.")
        return {"evidence": []}
    
    # Разбиваем тексты на чанки
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        if text.strip():
            chunks.extend(splitter.split_text(text))
    
    if not chunks:
        print("⚠️ Нет валидных чанков после разделения.")
        return {"evidence": []}
    
    print(f"📄 Подготовлено {len(chunks)} чанков для поиска")
    
    try:
        # Создаём векторное хранилище в памяти
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model
        )
        
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
    
    except Exception as e:
        print(f"❌ Ошибка при поиске доказательств: {e}")
        return {"evidence": [], "error": str(e)}