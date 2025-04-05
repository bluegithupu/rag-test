#!/usr/bin/env python
"""
Simple test script for the RAG system.
"""
import os
from app.document_loader import DocumentLoader
from app.processor import DocumentProcessor

def main():
    """
    Test the document loading and processing functionality.
    """
    print("Testing Document Loading and Processing...")
    
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
    
    # Print sample chunks
    print("\nSample document chunks:")
    for i, doc in enumerate(processed_documents[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
