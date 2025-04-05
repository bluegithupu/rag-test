"""
Embedding generation functionality for the RAG system.
"""
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings

class EmbeddingGenerator:
    """Generates embeddings for documents."""

    def __init__(self, model_name: str = settings.embedding_model, use_openai: bool = True):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the embedding model
            use_openai: Whether to use OpenAI embeddings
        """
        self.model_name = model_name
        self.use_openai = use_openai

        if use_openai:
            self.embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_base_url if settings.openai_base_url else None
            )
        else:
            # Use Sentence Transformers as a fallback
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        return self.embeddings.embed_documents(texts)

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        return self.embeddings.embed_query(query)
