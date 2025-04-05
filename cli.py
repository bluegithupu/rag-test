#!/usr/bin/env python
"""
Command-line interface for the RAG system.
"""
import argparse
import sys
from typing import List
from app.rag_pipeline import RAGPipeline

def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--source", required=True, help="Source directory or file")
    index_parser.add_argument("--recursive", action="store_true", help="Recursively search directories")
    
    # Index URLs command
    index_urls_parser = subparsers.add_parser("index-urls", help="Index documents from URLs")
    index_urls_parser.add_argument("--urls", required=True, nargs="+", help="URLs to index")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question to ask")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the document index")
    
    return parser.parse_args(args)

def main(args: List[str] = None) -> None:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command-line arguments
    """
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parse_args(args)
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    # Execute command
    if parsed_args.command == "index":
        num_docs = rag_pipeline.index_documents(
            source=parsed_args.source,
            recursive=parsed_args.recursive
        )
        print(f"Indexed {num_docs} documents")
    
    elif parsed_args.command == "index-urls":
        num_docs = rag_pipeline.index_urls(parsed_args.urls)
        print(f"Indexed {num_docs} documents from URLs")
    
    elif parsed_args.command == "query":
        answer = rag_pipeline.query(parsed_args.question)
        print("\nQuestion:", parsed_args.question)
        print("\nAnswer:", answer)
    
    elif parsed_args.command == "clear":
        rag_pipeline.clear_index()
        print("Index cleared successfully")
    
    else:
        print("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main()
