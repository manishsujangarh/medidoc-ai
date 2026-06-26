from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from app.agents.state import MediDocState
from app.agents.tools import search_document
from app.core.config import settings

llm = ChatGoogleGenerativeAI(
    model=settings.llm_model,
    google_api_key=settings.google_api_key,
    temperature=0.1
)

BILL_ANALYSER_PROMPT = """You are a medical billing expert helping US patients 
understand their hospital bills and EOBs.

When given a question:
1. Use the search_document tool to find the relevant charges
2. Explain what each charge means in plain English
3. Clarify the difference between billed amount, allowed amount, and patient balance
4. Flag anything that looks unusually high compared to typical rates
5. Always mention the page number where you found the information

Be warm, clear, and jargon-free. The patient may be stressed about this bill.
"""

def bill_analyser_node(state: MediDocState) -> MediDocState:
    agent = create_react_agent(llm, tools=[search_document], 
                                prompt=BILL_ANALYSER_PROMPT)
    doc_id = state.get("doc_id", "")
    messages = state["messages"]

    # Inject doc_id context into the last user message
    last_msg = messages[-1].content
    augmented = messages[:-1] + [
        type(messages[-1])(content=f"{last_msg}\n\n[doc_id: {doc_id}]")
    ]

    result = agent.invoke({"messages": augmented})
    answer = result["messages"][-1].content

    return {
        "messages": [AIMessage(content=answer, name="bill_analyser")],
        "findings": {**state.get("findings", {}), "bill_analysis": answer}
    }