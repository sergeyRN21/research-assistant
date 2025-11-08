# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from graph import app

st.set_page_config(page_title="🧠 Research Assistant", layout="wide")
st.title("🧠 Research Assistant — Научный ассистент с доказательствами")

st.markdown("""
Отвечает на научные вопросы, опираясь только на реальные статьи из arXiv.  
Никаких галлюцинаций. Только цитаты, проверки и прослеживаемость.
""")

question = st.text_input("Введите научный вопрос:", placeholder="Какие методы снижают KV-cache?")

if st.button("🔍 Запустить анализ"):
    if not question.strip():
        st.error("Введите вопрос!")
    else:
        # 🔥 Точная структура, как в твоём GraphState
        initial_state = {
            "question": question,
            "papers": [],
            "chunks_with_metadata": [],
            "hypotheses": [],
            "queries": [],  # ← добавлено
            "evidence": [],
            "final_answer": "",
            "retry_count": 0,
            "error": ""
        }

        status_text = st.empty()
        progress_bar = st.progress(0)

        def update_progress(step, total=6):
            progress = step / total
            progress_bar.progress(progress)
            status_text.text(f"🔄 Выполняется: {step}/6 — {step_names[step - 1]}")

        step_names = [
            "Поиск статей",
            "Извлечение текста",
            "Генерация гипотез",
            "Multi-query (русский)",
            "Поиск доказательств",
            "Синтез ответа"
        ]

        with st.spinner("🚀 Анализ выполняется..."):
            try:
                # Запускаем граф LangGraph
                final_state = app.invoke(
                    initial_state,
                    config={"recursion_limit": 10}
                )
                
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