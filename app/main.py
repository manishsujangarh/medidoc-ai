from fastapi import FastAPI
from app.api.routes.documents import router as doc_router
from app.api.routes.agents import router as agent_router 

app = FastAPI(title="MediDoc AI", version="0.2.0")
app.include_router(doc_router)
app.include_router(agent_router)   

@app.get("/health")
def health():
    return {"status": "ok"}