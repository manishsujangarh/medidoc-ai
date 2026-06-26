from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import MediDocState
from app.core.config import settings
import json

llm = ChatGoogleGenerativeAI(
    model=settings.llm_model,
    google_api_key=settings.google_api_key,
    temperature=0
)

SUPERVISOR_PROMPT = """You are a supervisor for a medical document AI system.
Based on the user's question AND the document type, decide which agent handles it.

Document types:
- medical_bill: itemized hospital/clinic charges
- denial_letter: insurance denial or adverse benefit determination  
- eob: explanation of benefits
- unknown: not yet identified

Agents:
- bill_analyser: explain charges, totals, line items (best for medical_bill, eob)
- error_checker: find billing errors, duplicates, upcoding (best for medical_bill, eob)
- appeal_drafter: write appeal letters, dispute denials (best for denial_letter)

If the document is a denial_letter and the user asks about charges or bills,
still route to appeal_drafter — there are no bill line items in a denial letter.

Respond ONLY with JSON: {"next": "agent_name"}
"""

def supervisor_node(state: MediDocState) -> MediDocState:
    last_message = state["messages"][-1].content
    doc_type     = state.get("doc_type", "unknown")

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"Document type: {doc_type}\nUser request: {last_message}")
    ])
    try:
        decision   = json.loads(response.content)
        next_agent = decision.get("next", "bill_analyser")
    except Exception:
        next_agent = "bill_analyser"

    return {"next": next_agent}