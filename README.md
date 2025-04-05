# RAG System MVP

A Retrieval-Augmented Generation (RAG) system implemented using LangChain and Python.

## Documentation

Please see the [documentation](doc/README.md) for more information.

## Tests

Run the tests with:

```bash
python -m tests.test_simple
python -m tests.test_vector_store
python -m tests.test_full_rag
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Start the server:
```bash
python server.py
```

4. Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

5. Access the web interface at http://localhost:8501
