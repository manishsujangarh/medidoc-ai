from langchain_core.tools import tool
from app.rag.retriever import retrieve

@tool
def search_document(query: str, doc_id: str) -> str:
    """
    Search the uploaded medical document for relevant information.
    Use this to find specific charges, denial reasons, CPT codes,
    dates, amounts, or any other details from the document.
    """
    chunks = retrieve(question=query, doc_id=doc_id)
    if not chunks:
        return "No relevant content found in the document."
    results = []
    for c in chunks:
        results.append(f"[Page {c['page']} | relevance {c['score']:.2f}]\n{c['text']}")
    return "\n\n---\n\n".join(results)