"""
Document loading functionality for the RAG system.
"""
import os
from typing import List, Dict, Any, Optional
from app.logger import get_logger, log_function_call
from app.config import settings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader
)
from langchain_core.documents import Document

class DocumentLoader:
    """Loads documents from various sources."""

    def __init__(self):
        """Initialize the document loader."""
        # 初始化日志记录器
        self.logger = get_logger(
            "rag_system.document_loader",
            level=settings.log_level,
            console_output=settings.log_to_console,
            file_output=settings.log_to_file
        )
        self.logger.info("初始化文档加载器")

        # Map file extensions to appropriate loaders
        self.loader_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
        }
        self.logger.debug(f"支持的文件类型: {list(self.loader_map.keys())}")

    @log_function_call(logger=get_logger("rag_system.document_loader.load_document"))
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document.

        Args:
            file_path: Path to the document

        Returns:
            List of Document objects
        """
        self.logger.info(f"加载文档: {file_path}")
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() not in self.loader_map:
            self.logger.error(f"不支持的文件类型: {file_extension}")
            raise ValueError(f"Unsupported file extension: {file_extension}")

        loader_class = self.loader_map[file_extension.lower()]
        self.logger.debug(f"使用加载器: {loader_class.__name__} 加载文件: {file_path}")
        loader = loader_class(file_path)

        docs = loader.load()
        self.logger.info(f"成功加载文档: {file_path}, 共 {len(docs)} 个文档对象")
        return docs

    @log_function_call(logger=get_logger("rag_system.document_loader.load_documents"))
    def load_documents(self, directory: str, recursive: bool = False) -> List[Document]:
        """
        Load all documents from a directory.

        Args:
            directory: Directory containing documents
            recursive: Whether to recursively search subdirectories

        Returns:
            List of Document objects
        """
        self.logger.info(f"从目录加载文档: {directory}, 递归: {recursive}")
        documents = []

        for dirpath, _, filenames in os.walk(directory):
            self.logger.debug(f"扫描目录: {dirpath}")
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                _, file_extension = os.path.splitext(file_path)

                if file_extension.lower() in self.loader_map:
                    try:
                        docs = self.load_document(file_path)
                        documents.extend(docs)
                    except Exception as e:
                        self.logger.error(f"加载文档出错 {file_path}: {str(e)}", exc_info=True)

            if not recursive:
                self.logger.debug("非递归模式, 只扫描顶层目录")
                break

        self.logger.info(f"共加载 {len(documents)} 个文档对象")
        return documents

    @log_function_call(logger=get_logger("rag_system.document_loader.load_from_web"))
    def load_from_web(self, urls: List[str]) -> List[Document]:
        """
        Load documents from web URLs.

        Args:
            urls: List of URLs to load

        Returns:
            List of Document objects
        """
        self.logger.info(f"从网页加载文档, URL 数量: {len(urls)}")
        self.logger.debug(f"URLs: {urls}")

        try:
            loader = WebBaseLoader(urls)
            docs = loader.load()
            self.logger.info(f"成功从网页加载 {len(docs)} 个文档")
            return docs
        except Exception as e:
            self.logger.error(f"从网页加载文档出错: {str(e)}", exc_info=True)
            raise
