from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from app.agents.graph import medidoc_graph
from pydantic import BaseModel

router = APIRouter(prefix="/agent", tags=["agent"])

class AgentRequest(BaseModel):
    question: str
    doc_id: str | None = None
    doc_type: str | None = None    # ← new, client sends this from upload response

def extract_text(content) -> str:
    """Handle both string and list content from Gemini"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)

@router.post("/ask")
async def agent_ask(req: AgentRequest):
    result = medidoc_graph.invoke({
        "messages": [HumanMessage(content=req.question)],
        "doc_id": req.doc_id,
        "findings": {},
        "next": ""
    })

    last_message = result["messages"][-1]
    return {
        "answer": extract_text(last_message.content),
        "agent":  getattr(last_message, "name", "unknown"),
        "findings": result.get("findings", {})
    }