from langgraph.graph import StateGraph, END
from app.agents.state import MediDocState
from app.agents.supervisor import supervisor_node
from app.agents.bill_analyser import bill_analyser_node
from app.agents.error_checker import error_checker_node
from app.agents.appeal_drafter import appeal_drafter_node

def route(state: MediDocState) -> str:
    """Edge function — reads state.next set by supervisor"""
    return state.get("next", "bill_analyser")

def build_graph():
    graph = StateGraph(MediDocState)

    # Add all nodes
    graph.add_node("supervisor",     supervisor_node)
    graph.add_node("bill_analyser",  bill_analyser_node)
    graph.add_node("error_checker",  error_checker_node)
    graph.add_node("appeal_drafter", appeal_drafter_node)

    # Entry point always goes to supervisor first
    graph.set_entry_point("supervisor")

    # Supervisor routes conditionally
    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "bill_analyser":  "bill_analyser",
            "error_checker":  "error_checker",
            "appeal_drafter": "appeal_drafter",
        }
    )

    # All agents go to END after finishing
    graph.add_edge("bill_analyser",  END)
    graph.add_edge("error_checker",  END)
    graph.add_edge("appeal_drafter", END)

    return graph.compile()

# Singleton — build once at startup
medidoc_graph = build_graph()