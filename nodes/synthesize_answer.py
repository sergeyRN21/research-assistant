# nodes/synthesize_answer.py
def synthesize_answer(state):
    """
    Узел: Формирует итоговый ответ на основе подтверждённых гипотез.
    """
    print("🧠 Узел: Синтез ответа...")

    evidence_list = state.get("evidence", [])

    if not evidence_list:
        print("⚠️ Нет доказательств для синтеза.")
        return {"final_answer": "❌ Нет данных для формирования ответа."}

    confirmed_hypotheses = []
    partial_hypotheses = []

    for item in evidence_list:
        hypothesis = item["hypothesis"]
        validated_chunks = item["validated_chunks"]

        confirmed = [vc for vc in validated_chunks if vc["judgment"]["confirmed"]]
        partial = [vc for vc in validated_chunks if not vc["judgment"]["confirmed"] and vc["judgment"]["partial"]]

        if confirmed:
            avg_confidence = sum(c["judgment"]["confidence"] for c in confirmed) / len(confirmed)
            sources = [c["text"][:300] + "..." for c in confirmed]
            confirmed_hypotheses.append({
                "hypothesis": hypothesis,
                "confidence": avg_confidence,
                "sources": sources
            })
        elif partial:
            partial_hypotheses.append({
                "hypothesis": hypothesis,
                "confidence": max(p["judgment"]["confidence"] for p in partial)
            })

    lines = ["\n📊 Ответ на основе научных данных:\n"]

    if confirmed_hypotheses:
        lines.append("✅ **Подтверждённые методы:**")
        for i, h in enumerate(sorted(confirmed_hypotheses, key=lambda x: -x["confidence"]), 1):
            lines.append(f"{i}. {h['hypothesis']}")
            lines.append(f"   • Уверенность: {h['confidence']:.2f}")
            lines.append(f"   • Подтверждено в: \"{h['sources'][0]}\"")

    if partial_hypotheses:
        lines.append("\n🟡 **Возможные, но слабо подтверждённые методы:**")
        for h in partial_hypotheses:
            lines.append(f"• {h['hypothesis']} (уверенность: {h['confidence']:.2f})")

    if not confirmed_hypotheses and not partial_hypotheses:
        lines.append("❌ Ни одна из гипотез не нашла подтверждения в найденных статьях.")

    final_answer = "\n".join(lines)
    print("✅ Ответ сформирован.")

    return {"final_answer": final_answer}