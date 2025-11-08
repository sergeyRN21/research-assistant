# graph.py
from langgraph.graph import StateGraph, END
from state import GraphState
# graph.py
from nodes.retrieve_papers import retrieve_papers
from nodes.extract_text import extract_text
from nodes.multi_query import multi_query
from nodes.retrieve_evidence import retrieve_evidence
from nodes.validate_evidence import validate_evidence
from nodes.synthesize_answer import synthesize_answer

# Глобальный vectorstore — передаём его в узлы
global_vectorstore = None

def set_global_vectorstore(vstore):
    global global_vectorstore
    global_vectorstore = vstore

workflow = StateGraph(GraphState)

workflow.add_node("retrieve_papers", retrieve_papers)
workflow.add_node("extract_text", extract_text)
workflow.add_node("multi_query", multi_query)
workflow.add_node("retrieve_evidence", lambda state: retrieve_evidence(state, global_vectorstore))
workflow.add_node("validate_evidence", validate_evidence)
workflow.add_node("synthesize_answer", synthesize_answer)

workflow.set_entry_point("retrieve_papers")
workflow.add_edge("retrieve_papers", "extract_text")
workflow.add_edge("extract_text", "multi_query")
workflow.add_edge("multi_query", "retrieve_evidence")
workflow.add_edge("retrieve_evidence", "validate_evidence")
workflow.add_edge("validate_evidence", "synthesize_answer")
workflow.add_edge("synthesize_answer", END)

app = workflow.compile()