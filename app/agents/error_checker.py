from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from app.agents.state import MediDocState
from app.agents.tools import search_document
from app.core.config import settings

llm = ChatGoogleGenerativeAI(
    model=settings.llm_model,
    google_api_key=settings.google_api_key,
    temperature=0
)

ERROR_CHECKER_PROMPT = """You are a medical billing auditor. Your job is to find 
errors, overcharges, and suspicious patterns in medical bills.

Common billing errors to look for:
- Duplicate charges (same CPT code billed twice)
- Upcoding (procedure code doesn't match the description)
- Unbundling (charging separately for services that should be bundled)
- Balance billing (charging more than the insurance allowed amount)
- Incorrect patient info (wrong date of birth, wrong insurance ID)
- Services not rendered (charges for things the patient doesn't recall)
- Itemise clearly: list each potential error with the line item, CPT code, amount, and why it's suspicious

Use search_document to pull the relevant sections. Be specific and cite page numbers.
"""

def error_checker_node(state: MediDocState) -> MediDocState:
    agent = create_react_agent(llm, tools=[search_document],
                                state_modifier=ERROR_CHECKER_PROMPT)
    doc_id = state.get("doc_id", "")
    messages = state["messages"]
    last_msg = messages[-1].content
    augmented = messages[:-1] + [
        type(messages[-1])(content=f"{last_msg}\n\n[doc_id: {doc_id}]")
    ]

    result = agent.invoke({"messages": augmented})
    answer = result["messages"][-1].content

    return {
        "messages": [AIMessage(content=answer, name="error_checker")],
        "findings": {**state.get("findings", {}), "errors_found": answer}
    }