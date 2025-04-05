"""
LLM interface for the RAG system.
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.config import settings
from app.logger import get_logger, log_function_call

class LLMInterface:
    """Interface for the language model."""

    def __init__(
        self,
        model_name: str = settings.llm_model,
        temperature: float = settings.llm_temperature
    ):
        """
        Initialize the LLM interface.

        Args:
            model_name: Name of the LLM model
            temperature: Temperature for generation
        """
        # 初始化日志记录器
        self.logger = get_logger(
            "rag_system.llm",
            level=settings.log_level,
            console_output=settings.log_to_console,
            file_output=settings.log_to_file
        )
        self.logger.info("初始化 LLM 接口")

        self.model_name = model_name
        self.temperature = temperature
        self.logger.debug(f"模型: {model_name}, 温度: {temperature}")

        # Initialize LLM
        try:
            self.logger.debug("初始化 LLM")
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_base_url if settings.openai_base_url else None
            )
            self.logger.debug("LLM 初始化成功")
        except Exception as e:
            self.logger.error(f"LLM 初始化出错: {str(e)}", exc_info=True)
            raise

        # Define prompt template for RAG
        try:
            self.logger.debug("创建 RAG 提示模板")
            self.rag_prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
            )
            self.logger.debug("RAG 提示模板创建成功")

            # Create LLM chain
            self.logger.debug("创建 LLM 链")
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.rag_prompt_template
            )
            self.logger.debug("LLM 链创建成功")
        except Exception as e:
            self.logger.error(f"创建提示模板或 LLM 链出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.llm.format_documents"))
    def format_documents(self, documents: List[Document]) -> str:
        """
        Format documents for the prompt.

        Args:
            documents: List of documents

        Returns:
            Formatted context string
        """
        self.logger.debug(f"格式化 {len(documents)} 个文档作为提示上下文")

        try:
            context = "\n\n".join([doc.page_content for doc in documents])
            self.logger.debug(f"格式化后的上下文长度: {len(context)} 字符")
            return context
        except Exception as e:
            self.logger.error(f"格式化文档出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.llm.generate_response"))
    def generate_response(
        self,
        question: str,
        documents: List[Document]
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            question: User question
            documents: Retrieved documents

        Returns:
            Generated response
        """
        self.logger.info(f"生成响应, 问题: '{question[:50]}{'...' if len(question) > 50 else ''}', 文档数量: {len(documents)}")

        try:
            # 格式化文档
            context = self.format_documents(documents)
            self.logger.debug(f"上下文长度: {len(context)} 字符")

            # 生成响应
            self.logger.debug("调用 LLM 生成响应")
            response = self.chain.run(
                context=context,
                question=question
            )

            self.logger.info(f"响应生成成功, 长度: {len(response)} 字符")
            self.logger.debug(f"响应内容: '{response[:100]}{'...' if len(response) > 100 else ''}")
            return response
        except Exception as e:
            self.logger.error(f"生成响应出错: {str(e)}", exc_info=True)
            raise
