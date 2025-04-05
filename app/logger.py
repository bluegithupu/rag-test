"""
日志配置模块，为 RAG 系统提供集中式的日志功能。
"""
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

# 创建日志目录
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 日志文件名格式：rag_system_YYYY-MM-DD.log
current_date = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = LOG_DIR / f"rag_system_{current_date}.log"

# 日志格式
CONSOLE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def get_logger(name, level="info", console_output=True, file_output=True):
    """
    获取配置好的日志记录器。

    Args:
        name: 日志记录器名称
        level: 日志级别 (debug, info, warning, error, critical)
        console_output: 是否输出到控制台
        file_output: 是否输出到文件

    Returns:
        配置好的日志记录器
    """
    # 获取日志级别
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免重复添加处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 禁止日志传播到父级记录器，防止重复日志
    logger.propagate = False

    # 添加控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 添加文件处理器
    if file_output:
        # 使用 RotatingFileHandler 进行日志轮转
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(FILE_LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

# 创建根日志记录器
root_logger = get_logger("rag_system")

def log_function_call(logger):
    """
    装饰器，用于记录函数调用的日志。

    Args:
        logger: 日志记录器

    Returns:
        装饰器函数
    """
    # 确保传入的 logger 禁止传播
    logger.propagate = False

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"调用函数 {func.__name__} 开始")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"调用函数 {func.__name__} 结束")
                return result
            except Exception as e:
                logger.error(f"调用函数 {func.__name__} 出错: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator
