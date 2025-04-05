# RAG 系统 MVP

使用 LangChain 和 Python 实现的检索增强生成 (RAG) 系统。

## 概述

本项目实现了检索增强生成 (RAG) 系统的最小可行产品 (MVP)，该系统通过在生成答案之前从知识库中检索相关信息来增强大型语言模型 (LLM) 的响应。它使用 LangChain 作为 RAG 管道，Chroma 作为向量数据库，以及 OpenAI 提供嵌入和 LLM 服务。

## 功能

- 从文件和网址加载文档
- 文档处理和分块
- 使用 Chroma 进行向量存储
- 基于相似度的检索
- 与 OpenAI LLMs 集成
- 命令行界面
- 简单的 Web API

## 安装

1. 克隆仓库或下载代码。

2. 创建虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # 在 Windows 上: venv\Scripts\activate
```

3. 安装依赖项：

```bash
pip install -r requirements.txt
```

4. 创建包含 OpenAI API 密钥的 `.env` 文件：

```bash
cp .env.example .env
# 编辑 .env 并添加您的 OpenAI API 密钥
```

## 使用方法

### 命令行界面

索引文档：

```bash
python cli.py index --source ./documents --recursive
```

索引网页：

```bash
python cli.py index-urls --urls https://example.com https://example.org
```

查询系统：

```bash
python cli.py query "什么是检索增强生成？"
```

清除索引：

```bash
python cli.py clear
```

### Web API

启动服务器：

```bash
python server.py
```

API 将在 http://localhost:8000 上可用。

API 端点：
- `POST /index`：从源索引文档
- `POST /index-urls`：从 URL 索引文档
- `POST /query`：查询系统
- `POST /clear`：清除文档索引

## 配置

配置选项在 `.env` 文件中可用：

- `OPENAI_API_KEY`：您的 OpenAI API 密钥
- `VECTOR_DB_PATH`：存储向量数据库的路径
- `LLM_MODEL`：要使用的 LLM 模型（例如，"gpt-3.5-turbo"）
- `LLM_TEMPERATURE`：生成的温度
- `EMBEDDING_MODEL`：要使用的嵌入模型
- `CHUNK_SIZE`：文本块的大小
- `CHUNK_OVERLAP`：块之间的重叠
- `NUM_DOCUMENTS`：要检索的文档数量

## 项目结构

```
rag-system/
├── app/
│   ├── __init__.py
│   ├── config.py           # 配置设置
│   ├── document_loader.py  # 文档加载功能
│   ├── processor.py        # 文档处理和分块
│   ├── embeddings.py       # 嵌入生成
│   ├── vector_store.py     # 向量数据库接口
│   ├── retriever.py        # 文档检索逻辑
│   ├── llm.py              # LLM 接口
│   ├── rag_pipeline.py     # 主 RAG 协调
│   └── api.py              # API 端点
├── cli.py                  # 命令行界面
├── server.py               # Web 服务器入口点
├── requirements.txt        # 项目依赖项
├── .env.example            # 环境变量示例
└── README.md               # 项目文档
```

## 许可证

MIT
