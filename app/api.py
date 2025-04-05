"""
API endpoints for the RAG system.
"""
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI(title="RAG System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Define request and response models
class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    source: str
    recursive: bool = False

class IndexURLsRequest(BaseModel):
    """Request model for indexing URLs."""
    urls: List[str]

class QueryRequest(BaseModel):
    """Request model for querying the system."""
    question: str
    filter: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str

# Define API endpoints
@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Status message
    """
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/index", response_model=Dict[str, int])
async def index_documents(request: IndexRequest):
    """
    Index documents from a source.

    Args:
        request: Index request

    Returns:
        Number of documents indexed
    """
    try:
        num_docs = rag_pipeline.index_documents(
            source=request.source,
            recursive=request.recursive
        )
        return {"num_documents": num_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index-urls", response_model=Dict[str, int])
async def index_urls(request: IndexURLsRequest):
    """
    Index documents from URLs.

    Args:
        request: Index URLs request

    Returns:
        Number of documents indexed
    """
    try:
        num_docs = rag_pipeline.index_urls(request.urls)
        return {"num_documents": num_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.

    Args:
        request: Query request

    Returns:
        Generated response
    """
    try:
        answer = rag_pipeline.query(
            question=request.question,
            filter=request.filter
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_index():
    """
    Clear the document index.

    Returns:
        Success message
    """
    try:
        rag_pipeline.clear_index()
        return {"message": "Index cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
