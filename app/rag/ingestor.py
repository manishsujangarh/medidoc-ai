import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
from pinecone import Pinecone
from app.core.config import settings
import hashlib, uuid

# Initialize clients
client = genai.Client(api_key=settings.google_api_key)
pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)

def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

def chunk_pages(pages: list[dict], doc_id: str, doc_type: str = "unknown") -> list[dict]:
    chunks = []
    for p in pages:
        splits = splitter.split_text(p["text"])
        for j, chunk_text in enumerate(splits):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    "doc_id":     doc_id,
                    "doc_type":   doc_type,    # ← stored in Pinecone
                    "page":       p["page"],
                    "chunk_index": j,
                }
            })
    return chunks

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Gemini batch embedding — structures input correctly and locks 768 dimensions"""
    texts = [c["text"] for c in chunks]
    
    # 1. Wrap each text cleanly into an individual Content structure
    contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
    
    # 2. Call the embedding client with explicit model config parameters
    result = client.models.embed_content(
        model=settings.embedding_model,  # "gemini-embedding-2"
        contents=contents,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768    # <-- CRITICAL: Forces 3072 down to 768
        )
    )
    
    # 3. Match 1-to-1 array responses back to chunks
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = result.embeddings[i].values
        
    return chunks

def upsert_to_pinecone(chunks: list[dict]):
    vectors = [
        {
            "id": c["id"],
            "values": c["embedding"],
            "metadata": {**c["metadata"], "text": c["text"]}
        }
        for c in chunks
    ]
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

def detect_document_type(text: str) -> str:
    """Quick keyword classifier — runs on first 1000 chars"""
    sample = text[:1000].lower()
    if any(k in sample for k in ["adverse benefit", "denial", "not medically necessary", 
                                   "denied", "notice of"]):
        return "denial_letter"
    if any(k in sample for k in ["explanation of benefits", "eob", "your benefits"]):
        return "eob"
    if any(k in sample for k in ["amount due", "patient statement", "itemized", 
                                   "cpt", "billed amount"]):
        return "medical_bill"
    return "unknown"

def ingest_document(pdf_bytes: bytes, filename: str) -> dict:
    doc_id = hashlib.md5(pdf_bytes).hexdigest()
    pages  = extract_text_from_pdf(pdf_bytes)

    # detect type from first page
    first_page_text = pages[0]["text"] if pages else ""
    doc_type = detect_document_type(first_page_text)

    chunks = chunk_pages(pages, doc_id, doc_type)   # pass type into metadata
    chunks = embed_chunks(chunks)
    upsert_to_pinecone(chunks)
    return {
        "doc_id":   doc_id,
        "filename": filename,
        "doc_type": doc_type,          # ← return this
        "pages":    len(pages),
        "chunks":   len(chunks),
        "status":   "indexed"
    }