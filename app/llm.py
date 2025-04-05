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
                input_variables=["context", "question", "references"],
                template="""You are a helpful AI assistant. Answer the question based on the provided context and include citations to the relevant sources.

Context:
{context}

References:
{references}

Question: {question}

Answer requirements:
1. Provide a comprehensive answer based on the context
2. Use [number] format to cite sources inline (e.g., [1], [2])
3. Include a "Sources:" section at the end listing all cited references
4. If the context doesn't contain enough information, acknowledge this fact

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

    @log_function_call(logger=get_logger("rag_system.llm.extract_references"))
    def extract_references(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract reference information from documents.

        Args:
            documents: List of documents

        Returns:
            List of reference dictionaries
        """
        self.logger.debug(f"从 {len(documents)} 个文档提取引用信息")

        try:
            references = []
            for i, doc in enumerate(documents):
                ref = {
                    "id": i + 1,
                    "source": doc.metadata.get("source", "Unknown"),
                    "title": doc.metadata.get("title", "Untitled Document"),
                }

                # 添加页码（如果有）
                if "page" in doc.metadata:
                    ref["page"] = doc.metadata["page"]

                # 添加 URL（如果有）
                if "url" in doc.metadata:
                    ref["url"] = doc.metadata["url"]

                # 添加块信息
                if "chunk_id" in doc.metadata:
                    ref["chunk_id"] = doc.metadata["chunk_id"]

                references.append(ref)

            self.logger.debug(f"提取了 {len(references)} 个引用")
            return references
        except Exception as e:
            self.logger.error(f"提取引用信息出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.llm.format_references"))
    def format_references(self, references: List[Dict[str, Any]]) -> str:
        """
        Format references for the prompt.

        Args:
            references: List of reference dictionaries

        Returns:
            Formatted references string
        """
        self.logger.debug(f"格式化 {len(references)} 个引用")

        try:
            formatted_refs = []
            for ref in references:
                ref_str = f"[{ref['id']}] "

                if "title" in ref:
                    ref_str += f"来源：《{ref['title']}》"

                if "page" in ref:
                    ref_str += f"，页码：{ref['page']}"

                if "url" in ref:
                    ref_str += f"，URL：{ref['url']}"

                formatted_refs.append(ref_str)

            result = "\n".join(formatted_refs)
            self.logger.debug(f"格式化后的引用长度: {len(result)} 字符")
            return result
        except Exception as e:
            self.logger.error(f"格式化引用出错: {str(e)}", exc_info=True)
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
            # 为每个文档添加引用标记
            formatted_docs = []
            for i, doc in enumerate(documents):
                doc_text = f"[{i+1}] {doc.page_content}"
                formatted_docs.append(doc_text)

            context = "\n\n".join(formatted_docs)
            self.logger.debug(f"格式化后的上下文长度: {len(context)} 字符")
            return context
        except Exception as e:
            self.logger.error(f"格式化文档出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.llm.generate_response_with_citations"))
    def generate_response_with_citations(
        self,
        question: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Generate a response with citations using the LLM.

        Args:
            question: User question
            documents: Retrieved documents

        Returns:
            Dictionary containing the answer and references
        """
        self.logger.info(f"生成带引用的响应, 问题: '{question[:50]}{'...' if len(question) > 50 else ''}', 文档数量: {len(documents)}")

        try:
            # 提取引用信息
            references = self.extract_references(documents)
            self.logger.debug(f"提取了 {len(references)} 个引用")

            # 格式化引用
            formatted_refs = self.format_references(references)
            self.logger.debug(f"引用格式化完成, 长度: {len(formatted_refs)} 字符")

            # 格式化文档
            context = self.format_documents(documents)
            self.logger.debug(f"上下文长度: {len(context)} 字符")

            # 生成响应
            self.logger.debug("调用 LLM 生成带引用的响应")
            response = self.chain.run(
                context=context,
                question=question,
                references=formatted_refs
            )

            self.logger.info(f"响应生成成功, 长度: {len(response)} 字符")
            self.logger.debug(f"响应内容: '{response[:100]}{'...' if len(response) > 100 else ''}")

            return {
                "answer": response,
                "references": references,
                "relevant_docs": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in documents
                ]
            }
        except Exception as e:
            self.logger.error(f"生成带引用的响应出错: {str(e)}", exc_info=True)
            raise

    @log_function_call(logger=get_logger("rag_system.llm.generate_response"))
    def generate_response(
        self,
        question: str,
        documents: List[Document]
    ) -> str:
        """
        Generate a response using the LLM (legacy method).

        Args:
            question: User question
            documents: Retrieved documents

        Returns:
            Generated response
        """
        self.logger.info(f"生成响应(旧方法), 问题: '{question[:50]}{'...' if len(question) > 50 else ''}', 文档数量: {len(documents)}")
        self.logger.warning("使用旧的生成响应方法，建议使用 generate_response_with_citations")

        try:
            # 调用新方法并只返回答案部分
            result = self.generate_response_with_citations(question, documents)
            return result["answer"]
        except Exception as e:
            self.logger.error(f"生成响应出错: {str(e)}", exc_info=True)
            raise
