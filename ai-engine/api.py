from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import ask_codemind  # Importing your existing function

# 1. Initialize the FastAPI app
app = FastAPI(title="CodeMind AI Engine", version="1.0")

# 2. Define the exact JSON structure we expect from the Go backend
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# 3. Create the API Endpoint
@app.post("/api/v1/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # Pass the question to your RAG loop
        ai_response = ask_codemind(request.question)
        
        # Return the answer as clean JSON
        return {"answer": ai_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Health Check Endpoint (Good practice for microservices)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CodeMind AI Engine"}
