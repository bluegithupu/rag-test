"""
Document retrieval logic for the RAG system.
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from app.vector_store import VectorStore
from app.config import settings

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
        self.vector_store = vector_store or VectorStore()
        self.num_documents = num_documents

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
        return self.vector_store.similarity_search(
            query=query,
            k=self.num_documents,
            filter=filter
        )

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
        return self.retrieve(query, filter)
