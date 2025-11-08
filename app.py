# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from graph import app
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🚀 Глобальное векторное хранилище — создаётся один раз
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = None  # Инициализируем позже

st.set_page_config(page_title="🧠 Research Assistant", layout="wide")
st.title("🧠 Research Assistant — Научный ассистент с доказательствами")

question = st.text_input("Введите научный вопрос:", placeholder="Какие методы снижают KV-cache?")

# app.py (внутри кнопки "Запустить анализ")
if st.button("🔍 Запустить анализ"):
    if not question.strip():
        st.error("Введите вопрос!")
    else:
        initial_state = {
            "question": question,
            "papers": [],
            "chunks_with_metadata": [],
            "hypotheses": [],
            "evidence": [],
            "final_answer": "",
            "retry_count": 0,
            "error": ""
        }

        with st.spinner("🚀 Анализ выполняется..."):
            try:
                # 1. Сначала получаем чанки
                result = app.invoke(initial_state, config={"recursion_limit": 10})
                chunks_data = result.get("chunks_with_metadata", [])

                # 2. Если есть чанки — создаем vectorstore
                if chunks_data:
                    texts = [chunk["text"] for chunk in chunks_data]
                    metadatas = [chunk.get("metadata", {}) for chunk in chunks_data]

                    # 🚀 Создаём векторное хранилище один раз
                    global_vectorstore = FAISS.from_texts(
                        texts=texts,
                        embedding=embedding_model,
                        metadatas=metadatas
                    )

                    # 3. Устанавливаем его в graph
                    from graph import set_global_vectorstore
                    set_global_vectorstore(global_vectorstore)

                    # 4. Повторно запускаем граф — теперь с vectorstore
                    final_state = app.invoke(result, config={"recursion_limit": 10})

                else:
                    final_state = result

                st.success("✅ Анализ завершён!")
                st.markdown("### 📝 Ответ")
                st.markdown(final_state["final_answer"])

                st.markdown("### 🔗 Цепочка доказательств")
                for item in final_state.get("evidence", []):
                    hyp = item.get("hypothesis", "Без названия")
                    with st.expander(f"Гипотеза: {hyp}"):
                        for vc in item.get("validated_chunks", []):
                            j = vc["judgment"]
                            status = "✅ Подтверждено" if j["confirmed"] else ("🟡 Частично" if j["partial"] else "❌ Не подтверждено")
                            st.markdown(f"""
                            - **Статус**: {status}
                            - **Уверенность**: {j['confidence']:.2f}
                            - **Причина**: {j['reason']}
                            - **Фрагмент**: *{vc['text'][:300]}...*
                            """)

            except Exception as e:
                st.error(f"❌ Ошибка: {e}")