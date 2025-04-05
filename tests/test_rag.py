#!/usr/bin/env python
"""
Test script for the RAG system.
"""
import os
import sys
from app.rag_pipeline import RAGPipeline

def main():
    """
    Test the RAG system by indexing a document and querying it.
    """
    print("Testing RAG System...")
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    # Index documents
    print("\nIndexing documents...")
    num_docs = rag_pipeline.index_documents("./data/documents", recursive=True)
    print(f"Indexed {num_docs} documents")
    
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
        answer = rag_pipeline.query(query)
        print(f"Answer: {answer}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
