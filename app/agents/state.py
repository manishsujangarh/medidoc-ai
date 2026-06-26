from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class MediDocState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
    doc_id: str | None  
    doc_type:  str | None                       # which document we're working on
    findings: dict                            # accumulated results from agents
    next: str                                 # which agent runs next