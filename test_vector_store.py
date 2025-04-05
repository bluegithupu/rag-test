#!/usr/bin/env python
"""
Test script for the vector store functionality.
"""
import os
import shutil
from app.document_loader import DocumentLoader
from app.processor import DocumentProcessor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def main():
    """
    Test the vector store functionality.
    """
    print("Testing Vector Store Functionality...")
    
    # Set up vector store directory
    vector_store_dir = "./data/test_vectordb"
    if os.path.exists(vector_store_dir):
        shutil.rmtree(vector_store_dir)
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Initialize components
    document_loader = DocumentLoader()
    document_processor = DocumentProcessor()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load documents
    print("\nLoading documents...")
    documents = document_loader.load_documents("./data/documents", recursive=True)
    print(f"Loaded {len(documents)} documents")
    
    # Process documents
    print("\nProcessing documents...")
    processed_documents = document_processor.process_documents(documents)
    print(f"Created {len(processed_documents)} document chunks")
    
    # Create vector store
    print("\nCreating vector store...")
    vector_store = Chroma.from_documents(
        documents=processed_documents,
        embedding=embeddings,
        persist_directory=vector_store_dir
    )
    print(f"Vector store created at {vector_store_dir}")
    
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
        
        # Print results
        print(f"Retrieved {len(docs)} relevant documents")
        for i, doc in enumerate(docs):
            print(f"\nResult {i+1}:")
            print(f"Content: {doc.page_content[:150]}...")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
