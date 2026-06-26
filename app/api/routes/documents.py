from fastapi import APIRouter, UploadFile, File, HTTPException
from google import genai
from google.genai import types
from app.rag.ingestor import ingest_document
from app.rag.retriever import retrieve
from app.rag.prompts import build_prompt
from app.core.config import settings

client = genai.Client(api_key=settings.google_api_key)

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    pdf_bytes = await file.read()
    return ingest_document(pdf_bytes, file.filename)

@router.post("/ask")
async def ask_question(question: str, doc_id: str | None = None):
    chunks = retrieve(question, doc_id)
    if not chunks:
        return {"answer": "No relevant content found. Please upload your document first."}
    
    system_prompt, user_prompt = build_prompt(question, chunks)
    
    # Fully updated syntax
    response = client.models.generate_content(
        model=settings.llm_model,      # Resolves to "gemini-2.5-flash"
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        )
    )
    
    return {
        "answer": response.text,
        "sources": [
            {"page": c["page"], "score": round(c["score"], 3)} for c in chunks
        ]
    }