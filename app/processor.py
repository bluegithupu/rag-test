"""
Document processing functionality for the RAG system.
"""
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.logger import get_logger, log_function_call

class DocumentProcessor:
    """Processes documents for the RAG system."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        # 初始化日志记录器
        self.logger = get_logger(
            "rag_system.processor",
            level=settings.log_level,
            console_output=settings.log_to_console,
            file_output=settings.log_to_file
        )
        self.logger.info("初始化文档处理器")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger.debug(f"块大小: {chunk_size}, 块重叠: {chunk_overlap}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.logger.debug("文本分割器初始化完成")

    @log_function_call(logger=get_logger("rag_system.processor.clean_text"))
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        self.logger.debug(f"清理文本, 原始长度: {len(text)}")

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove URLs (simplified)
        text = re.sub(r'https?://\S+', '', text)

        self.logger.debug(f"清理后文本长度: {len(text)}")
        return text

    def _enhance_metadata(self, metadata: Dict[str, Any], doc_index: int) -> Dict[str, Any]:
        """
        Enhance document metadata with additional information.

        Args:
            metadata: Original document metadata
            doc_index: Document index

        Returns:
            Enhanced metadata dictionary
        """
        # Create a copy of the original metadata
        enhanced = metadata.copy() if metadata else {}

        # Add document index
        enhanced["doc_index"] = doc_index
        enhanced["doc_id"] = f"doc_{doc_index}"

        # Add timestamp if not present
        if "created_at" not in enhanced:
            enhanced["created_at"] = datetime.now().isoformat()

        # Ensure source information is present
        if "source" not in enhanced:
            enhanced["source"] = "Unknown"

        # Extract title from filename if not present
        if "title" not in enhanced and "source" in enhanced:
            source = enhanced["source"]
            if os.path.isfile(source):
                enhanced["title"] = os.path.basename(source)
            else:
                enhanced["title"] = "Untitled Document"

        return enhanced

    @log_function_call(logger=get_logger("rag_system.processor.split_documents"))
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        self.logger.info(f"分割文档, 文档数量: {len(documents)}")

        try:
            split_docs = self.text_splitter.split_documents(documents)
            self.logger.info(f"文档分割完成, 生成块数量: {len(split_docs)}")
            return split_docs
        except Exception as e:
            self.logger.error(f"文档分割出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.processor.process_documents"))
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents: clean text and split into chunks.

        Args:
            documents: List of documents to process

        Returns:
            List of processed documents
        """
        self.logger.info(f"开始处理文档, 文档数量: {len(documents)}")

        # Clean text in documents
        self.logger.debug("清理文档文本")
        cleaned_documents = []
        for i, doc in enumerate(documents):
            self.logger.debug(f"清理文档 {i+1}/{len(documents)}")
            cleaned_text = self.clean_text(doc.page_content)

            # 增强文档元数据
            enhanced_metadata = self._enhance_metadata(doc.metadata, i)

            cleaned_doc = Document(
                page_content=cleaned_text,
                metadata=enhanced_metadata
            )
            cleaned_documents.append(cleaned_doc)

        self.logger.debug(f"清理完成, 开始分割文档")
        # Split documents into chunks
        split_docs = self.split_documents(cleaned_documents)

        # 为分割后的文档块添加块索引
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_id"] = f"chunk_{i}"

        self.logger.info(f"文档处理完成, 输入: {len(documents)} 文档, 输出: {len(split_docs)} 块")
        return split_docs
