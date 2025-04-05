"""
Main RAG orchestration for the RAG system.
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from app.document_loader import DocumentLoader
from app.processor import DocumentProcessor
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.llm import LLMInterface
from app.config import settings
from app.logger import get_logger, log_function_call

class RAGPipeline:
    """Main RAG orchestration."""

    def __init__(self):
        """Initialize the RAG pipeline."""
        # 初始化日志记录器
        self.logger = get_logger(
            "rag_system.pipeline",
            level=settings.log_level,
            console_output=settings.log_to_console,
            file_output=settings.log_to_file
        )
        self.logger.info("初始化 RAG 管道")

        # 初始化组件
        self.logger.debug("初始化文档加载器")
        self.document_loader = DocumentLoader()

        self.logger.debug("初始化文档处理器")
        self.document_processor = DocumentProcessor()

        self.logger.debug("初始化向量存储")
        self.vector_store = VectorStore()

        self.logger.debug("初始化检索器")
        self.retriever = Retriever(vector_store=self.vector_store)

        self.logger.debug("初始化 LLM 接口")
        self.llm_interface = LLMInterface()

        self.logger.info("RAG 管道初始化完成")

    @log_function_call(logger=get_logger("rag_system.pipeline.index_documents"))
    def index_documents(
        self,
        source: str,
        recursive: bool = False
    ) -> int:
        """
        Index documents from a source.

        Args:
            source: Source directory or file
            recursive: Whether to recursively search directories

        Returns:
            Number of documents indexed
        """
        self.logger.info(f"从源索引文档: {source}, 递归: {recursive}")

        try:
            # Load documents
            self.logger.debug(f"从 {source} 加载文档")
            documents = self.document_loader.load_documents(source, recursive)
            self.logger.info(f"加载了 {len(documents)} 个文档")

            # Process documents
            self.logger.debug("处理文档")
            processed_documents = self.document_processor.process_documents(documents)
            self.logger.info(f"处理后生成了 {len(processed_documents)} 个文档块")

            # Add to vector store
            self.logger.debug("将文档添加到向量存储")
            self.vector_store.add_documents(processed_documents)

            self.logger.info(f"成功索引了 {len(processed_documents)} 个文档块")
            return len(processed_documents)
        except Exception as e:
            self.logger.error(f"索引文档出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.pipeline.index_urls"))
    def index_urls(self, urls: List[str]) -> int:
        """
        Index documents from URLs.

        Args:
            urls: List of URLs

        Returns:
            Number of documents indexed
        """
        self.logger.info(f"从 URL 索引文档, URL 数量: {len(urls)}")
        self.logger.debug(f"URLs: {urls}")

        try:
            # Load documents
            self.logger.debug("从 Web 加载文档")
            documents = self.document_loader.load_from_web(urls)
            self.logger.info(f"从 Web 加载了 {len(documents)} 个文档")

            # Process documents
            self.logger.debug("处理文档")
            processed_documents = self.document_processor.process_documents(documents)
            self.logger.info(f"处理后生成了 {len(processed_documents)} 个文档块")

            # Add to vector store
            self.logger.debug("将文档添加到向量存储")
            self.vector_store.add_documents(processed_documents)

            self.logger.info(f"成功从 URL 索引了 {len(processed_documents)} 个文档块")
            return len(processed_documents)
        except Exception as e:
            self.logger.error(f"从 URL 索引文档出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.pipeline.query"))
    def query(
        self,
        question: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Query the RAG system.

        Args:
            question: User question
            filter: Metadata filter

        Returns:
            Generated response
        """
        self.logger.info(f"查询 RAG 系统, 问题: '{question[:50]}{'...' if len(question) > 50 else ''}")
        if filter:
            self.logger.debug(f"使用过滤器: {filter}")

        try:
            # Retrieve relevant documents
            self.logger.debug("检索相关文档")
            documents = self.retriever.get_relevant_documents(question, filter)
            self.logger.info(f"检索到 {len(documents)} 个相关文档")

            # Generate response
            self.logger.debug("生成响应")
            response = self.llm_interface.generate_response(question, documents)

            self.logger.info(f"生成响应成功, 长度: {len(response)} 字符")
            return response
        except Exception as e:
            self.logger.error(f"查询出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.pipeline.clear_index"))
    def clear_index(self) -> None:
        """Clear the document index."""
        self.logger.info("清除文档索引")

        try:
            self.vector_store.clear()
            self.logger.info("文档索引清除成功")
        except Exception as e:
            self.logger.error(f"清除文档索引出错: {str(e)}", exc_info=True)
            raise
