# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Убираем импорт utils/cache
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
        initial_state = {
            "question": question,
            "papers": [],
            "retrieved_texts": [],
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
                # Запускаем по шагам
                status_text.text("🔄 1/6 Поиск статей...")
                progress_bar.progress(0.1)
                state_step1 = retrieve_papers(initial_state)
                update_progress(1)

                status_text.text("🔄 2/6 Извлечение текста...")
                progress_bar.progress(0.2)
                state_step2 = extract_text(state_step1)
                update_progress(2)

                status_text.text("🔄 3/6 Генерация гипотез...")
                progress_bar.progress(0.4)
                state_step3 = generate_hypotheses(state_step2)
                update_progress(3)

                status_text.text("🔄 4/6 Поиск доказательств...")
                progress_bar.progress(0.6)
                state_step4 = retrieve_evidence(state_step3)
                update_progress(4)

                status_text.text("🔄 5/6 Валидация доказательств...")
                progress_bar.progress(0.8)
                state_step5 = validate_evidence(state_step4)
                update_progress(5)

                status_text.text("🔄 6/6 Синтез ответа...")
                progress_bar.progress(0.95)
                final_output = synthesize_answer(state_step5)
                update_progress(6)

                # Финальный результат
                st.success("✅ Анализ завершён!")
                st.markdown("### 📝 Ответ")
                st.markdown(final_output["final_answer"])

                # Цепочка доказательств
                st.markdown("### 🔗 Цепочка доказательств")
                evidence = state_step5.get("evidence", [])
                for item in evidence:
                    with st.expander(f"Гипотеза: {item['hypothesis']}"):
                        for vc in item["validated_chunks"]:
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