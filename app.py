# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from nodes.retrieve_papers import retrieve_papers
from nodes.extract_text import extract_text
from nodes.generate_hypotheses import generate_hypotheses
from nodes.retrieve_evidence import retrieve_evidence
from nodes.validate_evidence import validate_evidence
from nodes.synthesize_answer import synthesize_answer

st.set_page_config(page_title="🧠 Research Assistant", layout="wide")
st.title("🧠 Research Assistant — Научный ассистент с доказательствами")

st.markdown("""
Отвечает на научные вопросы, опираясь только на реальные статьи из arXiv.  
Никаких галлюцинаций. Только цитаты, проверки и прослеживаемость.
""")

question = st.text_input("Введите научный вопрос:", placeholder="Какие методы снижают KV-cache в LLM?")

if st.button("🔍 Запустить анализ"):
    if not question.strip():
        st.error("Введите вопрос!")
    else:
        # Начальное состояние
        state = {
            "question": question,
            "papers": [],
            "chunks_with_metadata": [],
            "hypotheses": [],
            "evidence": [],
            "final_answer": "",
            "retry_count": 0,
            "error": ""
        }

        # Показываем прогресс
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
            "Поиск доказательств",
            "Валидация доказательств",
            "Синтез ответа"
        ]

        with st.spinner("🚀 Запуск анализа..."):
            try:
                # === Шаг 1: Поиск статей ===
                status_text.text("🔄 1/6 Поиск статей...")
                progress_bar.progress(0.1)
                result = retrieve_papers(state)
                state.update(result)
                update_progress(1)

                # === Шаг 2: Извлечение текста ===
                status_text.text("🔄 2/6 Извлечение текста...")
                progress_bar.progress(0.2)
                result = extract_text(state)
                state.update(result)
                update_progress(2)

                # === Шаг 3: Генерация гипотез ===
                status_text.text("🔄 3/6 Генерация гипотез...")
                progress_bar.progress(0.4)
                result = generate_hypotheses(state)
                state.update(result)
                update_progress(3)

                # === Шаг 4: Поиск доказательств ===
                status_text.text("🔄 4/6 Поиск доказательств...")
                progress_bar.progress(0.6)
                result = retrieve_evidence(state)
                state.update(result)
                update_progress(4)

                # === Шаг 5: Валидация доказательств ===
                status_text.text("🔄 5/6 Валидация доказательств...")
                progress_bar.progress(0.8)
                result = validate_evidence(state)
                state.update(result)
                update_progress(5)

                # === Шаг 6: Синтез ответа ===
                status_text.text("🔄 6/6 Синтез ответа...")
                progress_bar.progress(0.95)
                result = synthesize_answer(state)
                state.update(result)
                update_progress(6)

                # === Финальный результат ===
                st.success("✅ Анализ завершён!")
                st.markdown("### 📝 Ответ")
                st.markdown(state["final_answer"])

                # === Цепочка доказательств ===
                st.markdown("### 🔗 Цепочка доказательств")
                evidence = state.get("evidence", [])
                for item in evidence:
                    hypothesis = item.get("hypothesis", "Без названия")
                    with st.expander(f"Гипотеза: {hypothesis}"):
                        for vc in item.get("validated_chunks", []):
                            j = vc.get("judgment", {})
                            status = "✅ Подтверждено" if j.get("confirmed") else ("🟡 Частично" if j.get("partial") else "❌ Не подтверждено")
                            confidence = j.get("confidence", 0)
                            reason = j.get("reason", "Нет объяснения")
                            text_snippet = vc.get("text", "")[:300] + "..." if len(vc.get("text", "")) > 300 else vc.get("text", "")
                            st.markdown(f"""
                            - **Статус**: {status}  
                            - **Уверенность**: {confidence:.2f}  
                            - **Причина**: {reason}  
                            - **Фрагмент**: *{text_snippet}*
                            """)

            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
                st.code(str(e))

# === Кнопка сброса ===
if st.button("🔄 Новый запрос"):
    for key in ["question", "papers", "chunks_with_metadata", "hypotheses", "evidence", "final_answer", "retry_count", "error"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()