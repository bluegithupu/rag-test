"""
Vector database interface for the RAG system.
"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.embeddings import EmbeddingGenerator
from app.config import settings

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
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.persist_directory = persist_directory

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize vector store
        self.vector_store = None

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        # If vector store exists, add documents
        if self.vector_store:
            self.vector_store.add_documents(documents)
        else:
            # Create new vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_generator.embeddings,
                persist_directory=self.persist_directory
            )

        # Persist vector store
        self.vector_store.persist()

    def load(self) -> None:
        """Load the vector store from disk."""
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                embedding_function=self.embedding_generator.embeddings,
                persist_directory=self.persist_directory
            )

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
        if not self.vector_store:
            self.load()

            if not self.vector_store:
                raise ValueError("Vector store not initialized")

        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )

    def clear(self) -> None:
        """Clear the vector store."""
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None
