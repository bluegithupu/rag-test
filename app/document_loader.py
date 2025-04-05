"""
Document loading functionality for the RAG system.
"""
import os
from typing import List, Dict, Any, Optional
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
        # Map file extensions to appropriate loaders
        self.loader_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
        }

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document.

        Args:
            file_path: Path to the document

        Returns:
            List of Document objects
        """
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() not in self.loader_map:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        loader_class = self.loader_map[file_extension.lower()]
        loader = loader_class(file_path)

        return loader.load()

    def load_documents(self, directory: str, recursive: bool = False) -> List[Document]:
        """
        Load all documents from a directory.

        Args:
            directory: Directory containing documents
            recursive: Whether to recursively search subdirectories

        Returns:
            List of Document objects
        """
        documents = []

        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                _, file_extension = os.path.splitext(file_path)

                if file_extension.lower() in self.loader_map:
                    try:
                        docs = self.load_document(file_path)
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

            if not recursive:
                break

        return documents

    def load_from_web(self, urls: List[str]) -> List[Document]:
        """
        Load documents from web URLs.

        Args:
            urls: List of URLs to load

        Returns:
            List of Document objects
        """
        loader = WebBaseLoader(urls)
        return loader.load()
