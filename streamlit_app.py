import streamlit as st
import requests
import json
import os
import tempfile
from pathlib import Path

# API 端点配置
API_URL = "http://localhost:8000"  # 根据您的实际部署情况修改

st.set_page_config(
    page_title="RAG 系统",
    page_icon="🔍",
    layout="wide"
)

st.title("📚 检索增强生成 (RAG) 系统")
st.markdown("使用这个界面来索引文档并查询信息。")

# 创建侧边栏
st.sidebar.header("RAG 系统设置")
api_url = st.sidebar.text_input("API URL", value=API_URL)

# 检查 API 连接
def check_api_connection(url):
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except:
        return False

api_status = check_api_connection(api_url)
if api_status:
    st.sidebar.success("✅ API 连接正常")
else:
    st.sidebar.error("❌ 无法连接到 API")
    st.sidebar.info("请确保 API 服务器已启动，并且 URL 正确")

# 创建标签页
tab1, tab2 = st.tabs(["查询", "文档管理"])

# 查询标签页
with tab1:
    st.header("查询信息")

    # 查询输入
    query = st.text_area("输入您的问题", height=100)

    # 查询选项
    show_citations = st.checkbox("显示引用信息", value=True, help="在回答中包含引用信息")
    show_sources = st.checkbox("显示原文", value=False, help="显示检索到的原文内容")

    # 查询按钮
    if st.button("提交查询", key="query_button", disabled=not api_status):
        if query:
            with st.spinner("正在生成回答..."):
                try:
                    # 调用查询 API
                    if show_citations:
                        # 使用带引用的 API
                        response = requests.post(
                            f"{api_url}/query-with-citations",
                            json={"question": query, "filter": None}
                        )
                    else:
                        # 使用普通 API
                        response = requests.post(
                            f"{api_url}/query",
                            json={"question": query, "filter": None}
                        )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("查询成功!")

                        # 显示回答
                        st.subheader("回答:")
                        st.markdown(result["answer"])

                        # 如果使用了带引用的 API，显示引用信息
                        if show_citations and "references" in result:
                            st.subheader("引用来源:")
                            references = result["references"]

                            for ref in references:
                                with st.expander(f"[{ref['id']}] {ref['title']}"):
                                    st.markdown(f"**来源:** {ref['source']}")
                                    if "page" in ref and ref["page"]:
                                        st.markdown(f"**页码:** {ref['page']}")
                                    if "url" in ref and ref["url"]:
                                        st.markdown(f"**URL:** [{ref['url']}]({ref['url']})")

                        # 如果选择显示原文，显示检索到的文档
                        if show_sources and "relevant_docs" in result:
                            st.subheader("检索到的原文:")
                            docs = result["relevant_docs"]

                            for i, doc in enumerate(docs):
                                with st.expander(f"文档 {i+1}"):
                                    st.markdown(f"```\n{doc['content']}\n```")
                                    st.markdown("**元数据:**")
                                    st.json(doc["metadata"])
                    else:
                        st.error(f"查询失败: {response.text}")
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")
        else:
            st.warning("请输入问题")

# 文档管理标签页
with tab2:
    st.header("文档管理")

    # 创建两列布局
    col1, col2 = st.columns(2)

    # 文件上传
    with col1:
        st.subheader("上传文档")
        uploaded_files = st.file_uploader(
            "选择文件上传",
            accept_multiple_files=True,
            type=["txt", "pdf", "docx", "html"]
        )

        if uploaded_files:
            if st.button("索引上传的文档", disabled=not api_status):
                # 创建临时目录存储上传的文件
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 保存上传的文件
                    file_paths = []
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(file_path)

                    # 显示进度
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 索引每个文件
                    for i, file_path in enumerate(file_paths):
                        status_text.text(f"正在索引: {os.path.basename(file_path)}")
                        try:
                            # 调用索引 API
                            response = requests.post(
                                f"{api_url}/index",
                                json={"source": file_path, "recursive": False}
                            )

                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"成功索引 {result['num_documents']} 个文档从 {os.path.basename(file_path)}")
                            else:
                                st.error(f"索引失败: {response.text}")
                        except Exception as e:
                            st.error(f"发生错误: {str(e)}")

                        # 更新进度
                        progress_bar.progress((i + 1) / len(file_paths))

                    status_text.text("索引完成!")

    # 目录索引
    with col2:
        st.subheader("索引目录")
        directory_path = st.text_input("输入目录路径")
        recursive = st.checkbox("递归索引子目录", value=True)

        if st.button("索引目录", disabled=not api_status):
            if directory_path:
                with st.spinner("正在索引目录..."):
                    try:
                        # 调用索引 API
                        response = requests.post(
                            f"{api_url}/index",
                            json={"source": directory_path, "recursive": recursive}
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"成功索引 {result['num_documents']} 个文档")
                        else:
                            st.error(f"索引失败: {response.text}")
                    except Exception as e:
                        st.error(f"发生错误: {str(e)}")
            else:
                st.warning("请输入目录路径")

    # 索引 URL
    st.subheader("索引网页")
    urls_input = st.text_area("输入 URL (每行一个)")

    if st.button("索引 URL", disabled=not api_status):
        if urls_input:
            urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
            if urls:
                with st.spinner("正在索引 URL..."):
                    try:
                        # 调用索引 URL API
                        response = requests.post(
                            f"{api_url}/index-urls",
                            json={"urls": urls}
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"成功索引 {result['num_documents']} 个文档")
                        else:
                            st.error(f"索引失败: {response.text}")
                    except Exception as e:
                        st.error(f"发生错误: {str(e)}")
            else:
                st.warning("请输入有效的 URL")

    # 清除索引
    st.subheader("管理索引")
    if st.button("清除所有索引", type="primary", help="这将删除所有已索引的文档", disabled=not api_status):
        with st.spinner("正在清除索引..."):
            try:
                # 调用清除 API
                response = requests.post(f"{api_url}/clear")

                if response.status_code == 200:
                    st.success("索引已清除")
                else:
                    st.error(f"清除索引失败: {response.text}")
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

# 添加页脚
st.markdown("---")
st.markdown("RAG 系统 | 基于 LangChain 和 Python 实现")

# 添加帮助信息
with st.sidebar.expander("使用帮助"):
    st.markdown("""
    ### 如何使用

    1. **查询**：在查询标签页输入问题并点击提交
       - 勾选“显示引用信息”可以查看回答的来源
       - 勾选“显示原文”可以查看检索到的原始文档
    2. **上传文档**：在文档管理标签页上传文件并索引
    3. **索引目录**：输入本地目录路径进行索引
    4. **索引网页**：输入网页 URL 进行索引
    5. **清除索引**：清除所有已索引的文档

    ### 引用功能

    系统现在支持引用功能，可以显示回答的来源信息：
    - 回答中的数字引用（如 [1], [2]）对应于引用来源部分的条目
    - 点击引用条目可以展开查看详细信息
    - “显示原文”选项可以查看完整的检索文档

    ### 注意事项

    - 确保 API 服务器已启动
    - 大文件上传可能需要较长时间
    - 支持的文件格式: TXT, PDF, DOCX, HTML
    """)

# 显示系统状态
with st.sidebar.expander("系统状态"):
    if api_status:
        st.markdown("- API 服务器: ✅ 在线")
    else:
        st.markdown("- API 服务器: ❌ 离线")
