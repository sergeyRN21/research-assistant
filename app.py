# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from graph import app

st.set_page_config(page_title="🧠 Research Assistant", layout="wide")
st.title("🧠 Research Assistant — Научный ассистент с доказательствами")

question = st.text_input("Введите научный вопрос:", placeholder="Какие методы снижают KV-cache?")

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
                # ⚡ Единственный вызов — LangGraph делает всё
                final_state = app.invoke(
                    initial_state,
                    config={"recursion_limit": 10}  # достаточно для 1 повтора
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