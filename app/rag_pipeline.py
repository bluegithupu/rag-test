"""
Main RAG orchestration for the RAG system.
"""
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from app.document_loader import DocumentLoader
from app.processor import DocumentProcessor
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.llm import LLMInterface

class RAGPipeline:
    """Main RAG orchestration."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.retriever = Retriever(vector_store=self.vector_store)
        self.llm_interface = LLMInterface()
    
    def index_documents(
        self,
        source: str,
        recursive: bool = False
    ) -> None:
        """
        Index documents from a source.
        
        Args:
            source: Source directory or file
            recursive: Whether to recursively search directories
        """
        # Load documents
        documents = self.document_loader.load_documents(source, recursive)
        
        # Process documents
        processed_documents = self.document_processor.process_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(processed_documents)
        
        return len(processed_documents)
    
    def index_urls(self, urls: List[str]) -> int:
        """
        Index documents from URLs.
        
        Args:
            urls: List of URLs
            
        Returns:
            Number of documents indexed
        """
        # Load documents
        documents = self.document_loader.load_from_web(urls)
        
        # Process documents
        processed_documents = self.document_processor.process_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(processed_documents)
        
        return len(processed_documents)
    
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
        # Retrieve relevant documents
        documents = self.retriever.get_relevant_documents(question, filter)
        
        # Generate response
        response = self.llm_interface.generate_response(question, documents)
        
        return response
    
    def clear_index(self) -> None:
        """Clear the document index."""
        self.vector_store.clear()
