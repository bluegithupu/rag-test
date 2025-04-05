"""
API endpoints for the RAG system.
"""
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline
from app.logger import get_logger
from app.config import settings

# 初始化日志记录器
api_logger = get_logger(
    "rag_system.api",
    level=settings.log_level,
    console_output=settings.log_to_console,
    file_output=settings.log_to_file
)

# Initialize FastAPI app
app = FastAPI(title="RAG System API")
api_logger.info("FastAPI 应用初始化")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)
api_logger.debug("CORS 中间件已配置")

# Initialize RAG pipeline
api_logger.info("初始化 RAG 管道")
rag_pipeline = RAGPipeline()
api_logger.info("RAG 管道初始化完成")

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

class ReferenceModel(BaseModel):
    """Model for a reference."""
    id: int
    source: str
    title: str
    page: Optional[int] = None
    url: Optional[str] = None
    chunk_id: Optional[str] = None

class DocumentModel(BaseModel):
    """Model for a document."""
    content: str
    metadata: Dict[str, Any]

class QueryResponseWithCitations(BaseModel):
    """Response model for queries with citations."""
    answer: str
    references: List[ReferenceModel]
    relevant_docs: List[DocumentModel]

class QueryResponse(BaseModel):
    """Response model for queries (legacy)."""
    answer: str

# Define API endpoints
@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Status message
    """
    api_logger.debug("健康检查端点被访问")
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
    api_logger.info(f"索引文档请求: 源={request.source}, 递归={request.recursive}")
    try:
        num_docs = rag_pipeline.index_documents(
            source=request.source,
            recursive=request.recursive
        )
        api_logger.info(f"成功索引 {num_docs} 个文档")
        return {"num_documents": num_docs}
    except Exception as e:
        api_logger.error(f"索引文档出错: {str(e)}", exc_info=True)
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
    api_logger.info(f"索引 URL 请求: URL 数量={len(request.urls)}")
    api_logger.debug(f"URLs: {request.urls}")
    try:
        num_docs = rag_pipeline.index_urls(request.urls)
        api_logger.info(f"成功从 URL 索引 {num_docs} 个文档")
        return {"num_documents": num_docs}
    except Exception as e:
        api_logger.error(f"索引 URL 出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-with-citations", response_model=QueryResponseWithCitations)
async def query_with_citations(request: QueryRequest):
    """
    Query the RAG system with citations.

    Args:
        request: Query request

    Returns:
        Generated response with citations
    """
    api_logger.info(f"查询请求(带引用): '{request.question[:50]}{'...' if len(request.question) > 50 else ''}")
    if request.filter:
        api_logger.debug(f"查询过滤器: {request.filter}")

    try:
        result = rag_pipeline.query_with_citations(
            question=request.question,
            filter=request.filter
        )
        api_logger.info(f"查询成功(带引用), 响应长度: {len(result['answer'])} 字符")
        return result
    except Exception as e:
        api_logger.error(f"查询出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system (legacy endpoint).

    Args:
        request: Query request

    Returns:
        Generated response
    """
    api_logger.info(f"查询请求(旧端点): '{request.question[:50]}{'...' if len(request.question) > 50 else ''}")
    api_logger.warning("使用旧的查询端点，建议使用 /query-with-citations")

    if request.filter:
        api_logger.debug(f"查询过滤器: {request.filter}")

    try:
        answer = rag_pipeline.query(
            question=request.question,
            filter=request.filter
        )
        api_logger.info(f"查询成功, 响应长度: {len(answer)} 字符")
        return {"answer": answer}
    except Exception as e:
        api_logger.error(f"查询出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_index():
    """
    Clear the document index.

    Returns:
        Success message
    """
    api_logger.info("清除索引请求")
    try:
        rag_pipeline.clear_index()
        api_logger.info("索引清除成功")
        return {"message": "Index cleared successfully"}
    except Exception as e:
        api_logger.error(f"清除索引出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
