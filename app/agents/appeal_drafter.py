from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from app.agents.state import MediDocState
from app.agents.tools import search_document
from app.core.config import settings

llm = ChatGoogleGenerativeAI(
    model=settings.llm_model,
    google_api_key=settings.google_api_key,
    temperature=0.3
)

APPEAL_DRAFTER_PROMPT = """You are an expert at writing insurance appeal letters 
for US patients. You help patients dispute denied claims professionally and effectively.

When drafting an appeal:
1. Use search_document to find the denial reason, claim number, dates, and amounts
2. Write a formal appeal letter with:
   - Patient name, member ID, claim number (from the document)
   - Clear statement of what is being appealed
   - Medical necessity argument (reference the clinical notes if available)
   - Cite relevant regulations (ACA external review rights, state insurance laws)
   - Request for expedited review if urgent
   - List of enclosed supporting documents
3. Keep the tone professional but firm
4. Include the appeal deadline from the denial letter
5. End with a clear call to action

Format the letter properly with date, addresses, salutation, body, and signature block.
"""

def appeal_drafter_node(state: MediDocState) -> MediDocState:
    agent = create_react_agent(llm, tools=[search_document],
                                state_modifier=APPEAL_DRAFTER_PROMPT)
    doc_id = state.get("doc_id", "")
    messages = state["messages"]
    last_msg = messages[-1].content
    augmented = messages[:-1] + [
        type(messages[-1])(content=f"{last_msg}\n\n[doc_id: {doc_id}]")
    ]

    result = agent.invoke({"messages": augmented})
    answer = result["messages"][-1].content

    return {
        "messages": [AIMessage(content=answer, name="appeal_drafter")],
        "findings": {**state.get("findings", {}), "appeal_letter": answer}
    }