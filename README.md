MediDoc AI

An AI-powered medical document intelligence system that helps US patients understand hospital bills, insurance EOBs, and denial letters — in plain English.

Built with Python, FastAPI, LangGraph, Google Gemini, and Pinecone. Designed as a production-grade portfolio project covering RAG pipelines, multi-agent workflows, voice AI, n8n automation, and AWS deployment.


The problem

100 million Americans carry medical debt. Most patients receive bills and denial letters they cannot understand — filled with CPT codes, contractual adjustments, and legal language. MediDoc AI reads these documents and answers questions in plain English, finds billing errors, and drafts appeal letters automatically.


Tech stack

LayerTechnologyAPIFastAPI + UvicornAI / LLMGoogle Gemini 1.5 FlashEmbeddingsGemini text-embedding-004Agent orchestrationLangGraphVector databasePineconePDF parsingPyMuPDFAutomationn8n (Module 4)DeploymentAWS EC2 + RDS + S3 (Module 5)Voice AIWhisper + ElevenLabs (Module 3)


Project structure

medidoc-ai/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py     # upload + simple RAG query
│   │       └── agents.py        # LangGraph agent endpoint
│   ├── rag/
│   │   ├── ingestor.py          # PDF parse, chunk, embed, store
│   │   ├── retriever.py         # semantic search
│   │   └── prompts.py           # system prompt templates
│   ├── agents/
│   │   ├── state.py             # shared LangGraph state
│   │   ├── supervisor.py        # routes to correct agent
│   │   ├── bill_analyser.py     # explains charges
│   │   ├── error_checker.py     # finds billing errors
│   │   ├── appeal_drafter.py    # writes appeal letters
│   │   ├── tools.py             # RAG as a LangGraph tool
│   │   └── graph.py             # wires the agent graph
│   └── core/
│       └── config.py            # environment settings
├── tests/
│   └── test_rag.py
├── .env
├── requirements.txt
└── README.md


Modules

ModuleWhat you buildKey skills1RAG pipelinePDF parsing, chunking, embeddings, Pinecone2Multi-agent systemLangGraph, supervisor routing, tool calling3Voice AIWhisper STT, ElevenLabs TTS4Automationn8n workflows, bill deadline alerts5AWS deploymentEC2, RDS, S3, CloudWatch, Docker6Portfolio packagingLoom demo, GitHub cleanup


Quickstart

1. Clone and set up

bashgit clone https://github.com/YOUR_USERNAME/medidoc-ai.git
cd medidoc-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Environment variables

Create a .env file in the project root:

bashGOOGLE_API_KEY=AIza...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=medidoc

Get your keys:


Google AI Studio: https://aistudio.google.com/app/apikey
Pinecone: https://app.pinecone.io


3. Create Pinecone index (run once)

bashpython -c "
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
pc.create_index(
    name='medidoc',
    dimension=768,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
print('Index created')
"

4. Run the server

bashuvicorn app.main:app --reload

Server runs at http://localhost:8000
Interactive API docs at http://localhost:8000/docs


API reference

Health check

GET /health

json{ "status": "ok" }


Upload a document

POST /documents/upload
Content-Type: multipart/form-data

bashcurl -X POST http://localhost:8000/documents/upload \
  -F "file=@medical_bill.pdf"

Response:

json{
  "doc_id": "abc123",
  "filename": "medical_bill.pdf",
  "doc_type": "medical_bill",
  "pages": 3,
  "chunks": 18,
  "status": "indexed"
}

Save the doc_id and doc_type — you need them for all queries.


Ask a question (simple RAG)

POST /documents/ask?question=...&doc_id=...

bashcurl -X POST "http://localhost:8000/documents/ask?question=What+is+my+total+amount+due&doc_id=abc123"

Response:

json{
  "answer": "Your total amount due is $1,269.00...",
  "sources": [{ "page": 2, "score": 0.91 }]
}


Ask the agent (multi-agent with auto-routing)

POST /agent/ask
Content-Type: application/json

bashcurl -X POST http://localhost:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Write me an appeal letter for my denied echocardiogram",
    "doc_id": "abc123",
    "doc_type": "denial_letter"
  }'

Response:

json{
  "answer": "Dear BlueCross BlueShield of Arizona...",
  "agent": "appeal_drafter",
  "findings": {
    "appeal_letter": "Dear BlueCross BlueShield..."
  }
}

The agent field tells you which specialist handled the request.


Agent routing logic

The supervisor reads both the user's question and the document type to route automatically:

Document typeQuestion typeAgent calledmedical_bill"what is this charge"bill_analysermedical_bill"any errors or duplicates"error_checkerdenial_letteranythingappeal_draftereob"explain my benefits"bill_analysereob"find mistakes"error_checker


Document types supported

TypeExamplesmedical_billHospital bill, clinic invoice, itemized statementdenial_letterInsurance denial, adverse benefit determinationeobExplanation of Benefits from insurer


Example questions to try

For a medical bill:


"What is my total amount due?"
"What does CPT code 99284 mean?"
"Why is the billed amount so different from what insurance paid?"
"Are there any duplicate charges in this bill?"
"What was I charged for on October 29?"


For a denial letter:


"Why was my claim denied?"
"What is the appeal deadline?"
"Write me a formal appeal letter"
"What documents do I need to appeal?"


For an EOB:


"What did my insurance actually pay?"
"What is my remaining deductible?"
"Is my out-of-pocket maximum close to being reached?"



Running tests

bashpytest tests/ -v


Requirements

fastapi
uvicorn
python-multipart
pymupdf
google-generativeai
langchain-google-genai
langchain
langchain-core
langgraph
pinecone-client
tiktoken
python-dotenv
pydantic-settings
pytest


Roadmap


 Module 1 — RAG pipeline
 Module 2 — LangGraph multi-agent system
 Module 3 — Voice AI (Whisper + ElevenLabs)
 Module 4 — n8n automation workflows
 Module 5 — AWS production deployment
 Module 6 — Portfolio cleanup and Loom demo



Author

Built as a portfolio project demonstrating production AI engineering — RAG, multi-agent orchestration, voice AI, automation, and cloud deployment using Python, FastAPI, LangGraph, and AWS.