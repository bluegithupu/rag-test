#!/usr/bin/env python
"""
Web server entry point for the RAG system.
"""
import uvicorn
from app.api import app
from app.logger import get_logger
from app.config import settings

# 初始化日志记录器
server_logger = get_logger(
    "rag_system.server",
    level=settings.log_level,
    console_output=settings.log_to_console,
    file_output=settings.log_to_file
)

def main():
    """Start the web server."""
    server_logger.info("启动 RAG 系统 Web 服务器")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        server_logger.error(f"启动服务器出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
