#!/usr/bin/env python
"""
Full test script for the RAG system using local models.
"""
import os
import shutil
from app.document_loader import DocumentLoader
from app.processor import DocumentProcessor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain

def main():
    """
    Test the full RAG system with local models.
    """
    print("Testing Full RAG System with Local Models...")

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

    # Define RAG prompt template
    rag_prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
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

        # Since we don't have a local LLM running, we'll just show what would be sent to the LLM
        print(f"Prompt that would be sent to LLM:")
        print(rag_prompt_template.format(context=context[:500] + "...", question=query))

    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
