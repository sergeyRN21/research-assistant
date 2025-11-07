# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Загружаем узлы
from nodes.retrieve_papers import retrieve_papers
from nodes.extract_text import extract_text
from nodes.generate_hypotheses import generate_hypotheses
from nodes.retrieve_evidence import retrieve_evidence
from nodes.validate_evidence import validate_evidence
from nodes.synthesize_answer import synthesize_answer

# Импортируем граф
from graph import app as research_app

st.set_page_config(page_title="🧠 Research Assistant", layout="wide")
st.title("🧠 Research Assistant — Научный ассистент с доказательствами")

st.markdown("""
Отвечает на научные вопросы, опираясь только на реальные статьи из arXiv.  
Никаких галлюцинаций. Только цитаты, проверки и прослеживаемость.
""")

# Ввод вопроса
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

        with st.spinner("🔎 Поиск статей..."):
            pass  # Индикатор будет обновляться ниже

        # Показ логов по ходу выполнения
        log_placeholder = st.empty()
        logs = []

        try:
            # Обработка через LangGraph (end-to-end)
            for output in research_app.stream(initial_state):
                step = list(output.keys())[0]
                logs.append(f"✅ {step}: Выполнено")
                log_placeholder.text_area("Лог выполнения", value="\n".join(logs), height=200)

            # Извлечение результата
            final_output = list(research_app.stream(initial_state))[-1]
            result = final_output.get("synthesize_answer", {}).get("final_answer", "❌ Ответ не сформирован.")

            # Вывод результата
            st.markdown("### 📝 Ответ")
            st.markdown(result)

            # Показать цепочку доказательств
            st.markdown("### 🔗 Цепочка доказательств")
            evidence = final_output.get("validate_evidence", {}).get("evidence", [])
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
            st.error(f"❌ Ошибка при выполнении: {e}")