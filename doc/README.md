# RAG System MVP

A Retrieval-Augmented Generation (RAG) system implemented using LangChain and Python.

## Overview

This project implements a Minimum Viable Product (MVP) of a RAG system that enhances Large Language Model (LLM) responses by retrieving relevant information from a knowledge base before generating answers. It uses LangChain for the RAG pipeline, Chroma as the vector database, and OpenAI for embeddings and LLM.

## Features

- Document loading from files and web URLs
- Document processing and chunking
- Vector storage with Chroma
- Similarity-based retrieval
- Integration with OpenAI LLMs
- Command-line interface
- Simple web API

## Installation

1. Clone the repository or download the code.

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Command-line Interface

Index documents:

```bash
python cli.py index --source ./documents --recursive
```

Index web pages:

```bash
python cli.py index-urls --urls https://example.com https://example.org
```

Query the system:

```bash
python cli.py query "What is retrieval-augmented generation?"
```

Clear the index:

```bash
python cli.py clear
```

### Web API

Start the server:

```bash
python server.py
```

The API will be available at http://localhost:8000.

API endpoints:
- `POST /index`: Index documents from a source
- `POST /index-urls`: Index documents from URLs
- `POST /query`: Query the system
- `POST /clear`: Clear the document index

## Configuration

Configuration options are available in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `VECTOR_DB_PATH`: Path to store the vector database
- `LLM_MODEL`: LLM model to use (e.g., "gpt-3.5-turbo")
- `LLM_TEMPERATURE`: Temperature for generation
- `EMBEDDING_MODEL`: Embedding model to use
- `CHUNK_SIZE`: Size of text chunks
- `CHUNK_OVERLAP`: Overlap between chunks
- `NUM_DOCUMENTS`: Number of documents to retrieve
- `LOG_LEVEL`: Logging level (debug, info, warning, error, critical)
- `LOG_TO_CONSOLE`: Whether to log to console (true/false)
- `LOG_TO_FILE`: Whether to log to file (true/false)

## Logging System

The system includes a comprehensive logging system for debugging and monitoring:

- Logs are stored in the `logs` directory with the format `rag_system_YYYY-MM-DD.log`
- Log rotation is implemented to prevent log files from growing too large
- Different log levels are supported: debug, info, warning, error, critical
- Each component has its own logger for better organization
- Function call tracing is available for debugging

To view logs:

```bash
# View the latest log file
cat logs/rag_system_$(date +%Y-%m-%d).log

# Follow log updates in real-time
tail -f logs/rag_system_$(date +%Y-%m-%d).log
```

## Project Structure

```
rag-system/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── document_loader.py  # Document loading functionality
│   ├── processor.py        # Document processing and chunking
│   ├── embeddings.py       # Embedding generation
│   ├── vector_store.py     # Vector database interface
│   ├── retriever.py        # Document retrieval logic
│   ├── llm.py              # LLM interface
│   ├── rag_pipeline.py     # Main RAG orchestration
│   ├── logger.py           # Logging system
│   └── api.py              # API endpoints
├── cli.py                  # Command-line interface
├── server.py               # Web server entry point
├── requirements.txt        # Project dependencies
├── .env.example            # Example environment variables
├── logs/                   # Log files directory
└── doc/                    # Documentation
```

## License

MIT
