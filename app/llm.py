"""
LLM interface for the RAG system.
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.config import settings

class LLMInterface:
    """Interface for the language model."""

    def __init__(
        self,
        model_name: str = settings.llm_model,
        temperature: float = settings.llm_temperature
    ):
        """
        Initialize the LLM interface.

        Args:
            model_name: Name of the LLM model
            temperature: Temperature for generation
        """
        self.model_name = model_name
        self.temperature = temperature

        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url if settings.openai_base_url else None
        )

        # Define prompt template for RAG
        self.rag_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        )

        # Create LLM chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.rag_prompt_template
        )

    def format_documents(self, documents: List[Document]) -> str:
        """
        Format documents for the prompt.

        Args:
            documents: List of documents

        Returns:
            Formatted context string
        """
        return "\n\n".join([doc.page_content for doc in documents])

    def generate_response(
        self,
        question: str,
        documents: List[Document]
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            question: User question
            documents: Retrieved documents

        Returns:
            Generated response
        """
        context = self.format_documents(documents)

        response = self.chain.run(
            context=context,
            question=question
        )

        return response
