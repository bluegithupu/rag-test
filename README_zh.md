# RAG 系统 MVP

使用 LangChain 和 Python 实现的检索增强生成 (RAG) 系统。

## 文档

请查看[文档](doc/README_zh.md)获取更多信息。

## 测试

运行测试：

```bash
python -m tests.test_simple
python -m tests.test_vector_store
python -m tests.test_full_rag
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件设置您的配置
```

3. 启动服务器：
```bash
python server.py
```

4. 启动 Streamlit 界面：
```bash
streamlit run streamlit_app.py
```

5. 在浏览器中访问 http://localhost:8501
