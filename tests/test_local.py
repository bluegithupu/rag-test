#!/usr/bin/env python
"""
Test script for the RAG system using local models.
"""
import os
from app.document_loader import DocumentLoader
from app.processor import DocumentProcessor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline

def main():
    """
    Test the RAG system with local models.
    """
    print("Testing RAG System with Local Models...")
    
    # Initialize components
    document_loader = DocumentLoader()
    document_processor = DocumentProcessor()
    
    # Load documents
    print("\nLoading documents...")
    documents = document_loader.load_documents("./data/documents", recursive=True)
    print(f"Loaded {len(documents)} documents")
    
    # Process documents
    print("\nProcessing documents...")
    processed_documents = document_processor.process_documents(documents)
    print(f"Created {len(processed_documents)} document chunks")
    
    # Initialize embeddings
    print("\nInitializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    print("\nCreating vector store...")
    vector_store = Chroma.from_documents(
        documents=processed_documents,
        embedding=embeddings,
        persist_directory="./data/vectordb"
    )
    
    # Test queries
    test_queries = [
        "What is RAG?",
        "What are the key components of a RAG system?",
        "What are the benefits of using RAG?",
        "What are some challenges in RAG systems?"
    ]
    
    print("\nTesting queries:")
    for query in test_queries:
        print(f"\nQuestion: {query}")
        # Retrieve relevant documents
        docs = vector_store.similarity_search(query, k=2)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        print(f"Retrieved {len(docs)} relevant documents")
        print(f"Context: {context[:200]}...")
        
        # In a real implementation, we would use the context with an LLM to generate an answer
        # For this test, we'll just show the retrieved context
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
