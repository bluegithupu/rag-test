"""
Document retrieval logic for the RAG system.
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from app.vector_store import VectorStore
from app.config import settings
from app.logger import get_logger, log_function_call

class Retriever:
    """Retrieves relevant documents for queries."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        num_documents: int = settings.num_documents
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store
            num_documents: Number of documents to retrieve
        """
        # 初始化日志记录器
        self.logger = get_logger(
            "rag_system.retriever",
            level=settings.log_level,
            console_output=settings.log_to_console,
            file_output=settings.log_to_file
        )
        self.logger.info("初始化检索器")

        self.vector_store = vector_store or VectorStore()
        self.num_documents = num_documents
        self.logger.debug(f"要检索的文档数量: {num_documents}")

    @log_function_call(logger=get_logger("rag_system.retriever.retrieve"))
    def retrieve(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            filter: Metadata filter

        Returns:
            List of relevant documents
        """
        self.logger.info(f"检索相关文档, 查询: '{query[:50]}{'...' if len(query) > 50 else ''}")
        if filter:
            self.logger.debug(f"使用过滤器: {filter}")

        try:
            self.logger.debug(f"使用相似度搜索, k={self.num_documents}")
            documents = self.vector_store.similarity_search(
                query=query,
                k=self.num_documents,
                filter=filter
            )

            self.logger.info(f"检索到 {len(documents)} 个相关文档")
            return documents
        except Exception as e:
            self.logger.error(f"检索相关文档出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.retriever.get_relevant_documents"))
    def get_relevant_documents(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Get relevant documents for a query.

        Args:
            query: Query text
            filter: Metadata filter

        Returns:
            List of relevant documents
        """
        self.logger.debug(f"获取相关文档, 查询: '{query[:50]}{'...' if len(query) > 50 else ''}")
        return self.retrieve(query, filter)
