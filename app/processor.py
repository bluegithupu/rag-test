"""
Document processing functionality for the RAG system.
"""
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove URLs (simplified)
        text = re.sub(r'https?://\S+', '', text)

        return text

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        return self.text_splitter.split_documents(documents)

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents: clean text and split into chunks.

        Args:
            documents: List of documents to process

        Returns:
            List of processed documents
        """
        # Clean text in documents
        cleaned_documents = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            cleaned_doc = Document(
                page_content=cleaned_text,
                metadata=doc.metadata
            )
            cleaned_documents.append(cleaned_doc)

        # Split documents into chunks
        split_docs = self.split_documents(cleaned_documents)

        return split_docs
