# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Импортируй узлы
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

# Кнопка запуска
if st.button("🔍 Запустить анализ") and question.strip():
    st.session_state.question = question
    st.session_state.step = 1
    st.session_state.state = {
        "question": question,
        "papers": [],
        "retrieved_texts": [],
        "hypotheses": [],
        "evidence": [],
        "final_answer": "",
        "retry_count": 0,
        "error": ""
    }

# Список шагов для отображения
step_names = [
    "Поиск статей",
    "Извлечение текста",
    "Генерация гипотез",
    "Поиск доказательств",
    "Валидация доказательств",
    "Синтез ответа"
]

# Показываем прогресс, если есть активный процесс
if 'step' in st.session_state and st.session_state.step <= len(step_names):
    current_step = st.session_state.step
    state = st.session_state.state

    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(1, current_step):
        status_text.text(f"✅ {i}/6 — {step_names[i-1]}")
        progress_bar.progress(i / 6)

    status_text.text(f"🔄 {current_step}/6 — {step_names[current_step-1]}")

    # === Шаг 1: Поиск статей ===
    if current_step == 1:
        updated = retrieve_papers(state)
        st.session_state.state.update(updated)
        st.session_state.step = 2
        st.rerun()

    # === Шаг 2: Извлечение текста ===
    elif current_step == 2:
        updated = extract_text(state)
        st.session_state.state.update(updated)
        st.session_state.step = 3
        st.rerun()

    # === Шаг 3: Генерация гипотез ===
    elif current_step == 3:
        updated = generate_hypotheses(state)
        st.session_state.state.update(updated)
        st.session_state.step = 4
        st.rerun()

    # === Шаг 4: Поиск доказательств ===
    elif current_step == 4:
        updated = retrieve_evidence(state)
        st.session_state.state.update(updated)
        st.session_state.step = 5
        st.rerun()

    # === Шаг 5: Валидация доказательств ===
    elif current_step == 5:
        updated = validate_evidence(state)
        st.session_state.state.update(updated)
        st.session_state.step = 6
        st.rerun()

    # === Шаг 6: Синтез ответа ===
    elif current_step == 6:
        updated = synthesize_answer(state)
        st.session_state.state.update(updated)
        st.session_state.step = 7  # завершён
        st.rerun()

# === Финальный вывод ===
if 'step' in st.session_state and st.session_state.step == 7:
    final_output = st.session_state.state
    st.success("✅ Анализ завершён!")
    
    st.markdown("### 📝 Ответ")
    st.markdown(final_output["final_answer"])

    st.markdown("### 🔗 Цепочка доказательств")
    evidence = final_output.get("evidence", [])
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

# Очистка (опционально)
if st.button("🔄 Новый запрос"):
    if 'step' in st.session_state:
        del st.session_state.step
        del st.session_state.state
        del st.session_state.question
    st.rerun()