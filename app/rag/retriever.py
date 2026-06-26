# 1. Switch to the modern SDK package imports
from google import genai
from google.genai import types
from pinecone import Pinecone
from app.core.config import settings

# 2. Initialize the modern GenAI Client
client = genai.Client(api_key=settings.google_api_key)
pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index)

def embed_query(question: str) -> list[float]:
    """Generates a query embedding forced down to 768 dimensions"""
    # 3. Use client.models.embed_content with proper config mappings
    result = client.models.embed_content(
        model=settings.embedding_model,  # e.g., "gemini-embedding-2"
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY", # Task types are now uppercase strings
            output_dimensionality=768    # <-- Proper way to squeeze vectors to 768
        )
    )
    # 4. Access vectors via object notation (.embeddings) instead of dict keys
    return result.embeddings[0].values

def retrieve(question: str, doc_id: str | None = None) -> list[dict]:
    query_vec = embed_query(question)
    filter_dict = {"doc_id": {"$eq": doc_id}} if doc_id else {}
    results = index.query(
        vector=query_vec,
        top_k=settings.top_k,
        include_metadata=True,
        filter=filter_dict or None
    )
    return [
        {
            "text": m.metadata["text"],
            "score": m.score,
            "page": m.metadata.get("page"),
            "doc_id": m.metadata.get("doc_id"),
        }
        for m in results.matches
    ]