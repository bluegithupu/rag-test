"""
Vector database interface for the RAG system.
"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.embeddings import EmbeddingGenerator
from app.config import settings
from app.logger import get_logger, log_function_call

class VectorStore:
    """Interface for the vector database."""

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        persist_directory: str = settings.vector_db_path
    ):
        """
        Initialize the vector store.

        Args:
            embedding_generator: Embedding generator
            persist_directory: Directory to persist the vector store
        """
        # 初始化日志记录器
        self.logger = get_logger(
            "rag_system.vector_store",
            level=settings.log_level,
            console_output=settings.log_to_console,
            file_output=settings.log_to_file
        )
        self.logger.info("初始化向量存储")

        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.persist_directory = persist_directory
        self.logger.debug(f"向量存储目录: {persist_directory}")

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        self.logger.debug(f"确保向量存储目录存在")

        # Initialize vector store
        self.vector_store = None
        self.logger.debug("向量存储初始化为 None, 将在需要时加载")

    @log_function_call(logger=get_logger("rag_system.vector_store.add_documents"))
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        self.logger.info(f"添加文档到向量存储, 文档数量: {len(documents)}")

        try:
            # If vector store exists, add documents
            if self.vector_store:
                self.logger.debug("向量存储已存在, 添加文档")
                self.vector_store.add_documents(documents)
            else:
                # Create new vector store
                self.logger.debug("创建新的向量存储")
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_generator.embeddings,
                    persist_directory=self.persist_directory
                )

            # Persist vector store
            self.logger.debug("持久化向量存储")
            self.vector_store.persist()
            self.logger.info("文档添加并持久化成功")
        except Exception as e:
            self.logger.error(f"添加文档到向量存储出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.vector_store.load"))
    def load(self) -> None:
        """Load the vector store from disk."""
        self.logger.info(f"从磁盘加载向量存储: {self.persist_directory}")

        if os.path.exists(self.persist_directory):
            try:
                self.logger.debug("向量存储目录存在, 开始加载")
                self.vector_store = Chroma(
                    embedding_function=self.embedding_generator.embeddings,
                    persist_directory=self.persist_directory
                )
                self.logger.info("向量存储加载成功")
            except Exception as e:
                self.logger.error(f"加载向量存储出错: {str(e)}", exc_info=True)
                raise
        else:
            self.logger.warning(f"向量存储目录不存在: {self.persist_directory}")

    @log_function_call(logger=get_logger("rag_system.vector_store.similarity_search"))
    def similarity_search(
        self,
        query: str,
        k: int = settings.num_documents,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Query text
            k: Number of documents to retrieve
            filter: Metadata filter

        Returns:
            List of similar documents
        """
        self.logger.info(f"执行相似度搜索, 查询: '{query[:50]}{'...' if len(query) > 50 else ''}', k={k}")
        if filter:
            self.logger.debug(f"使用过滤器: {filter}")

        try:
            if not self.vector_store:
                self.logger.debug("向量存储未初始化, 尝试加载")
                self.load()

                if not self.vector_store:
                    self.logger.error("向量存储未初始化且无法加载")
                    raise ValueError("Vector store not initialized")

            self.logger.debug(f"执行相似度搜索, k={k}")
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )

            self.logger.info(f"相似度搜索完成, 找到 {len(results)} 个结果")
            return results
        except Exception as e:
            self.logger.error(f"相似度搜索出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.vector_store.clear"))
    def clear(self) -> None:
        """Clear the vector store."""
        self.logger.info("清除向量存储")

        try:
            if self.vector_store:
                self.logger.debug("删除向量存储集合")
                self.vector_store.delete_collection()
                self.vector_store = None
                self.logger.info("向量存储清除成功")
            else:
                self.logger.warning("向量存储未初始化, 无需清除")
        except Exception as e:
            self.logger.error(f"清除向量存储出错: {str(e)}", exc_info=True)
            raise
