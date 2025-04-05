#!/usr/bin/env python
"""
Web server entry point for the RAG system.
"""
import uvicorn
from app.api import app

def main():
    """Start the web server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
