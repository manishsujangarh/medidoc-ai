MEDICAL_BILL_SYSTEM = """
You are MediDoc AI, an expert at explaining US medical bills, EOBs,
and insurance denial letters to patients in plain English.

Rules:
- Never give medical advice, only financial and administrative guidance
- Always cite the specific line item or section from the document
- If you spot a potential billing error, flag it clearly
- Suggest next steps (appeal, call insurer, payment plan)
- Be warm, clear, and jargon-free
"""

def build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    context = "\n\n---\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in chunks
    )
    user_prompt = f"""
Document excerpts:
{context}

Patient question: {question}
"""
    return MEDICAL_BILL_SYSTEM, user_prompt