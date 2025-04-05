#!/bin/bash

# 安装 Streamlit 和其他依赖的脚本

echo "正在设置 Streamlit 界面..."

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "警告: 未检测到虚拟环境。建议在虚拟环境中安装依赖。"
    echo "您可以使用以下命令创建并激活虚拟环境:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate  # 在 Windows 上使用: venv\\Scripts\\activate"
    
    read -p "是否继续安装? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "安装已取消"
        exit 1
    fi
fi

# 安装依赖
echo "正在安装 Streamlit 和其他依赖..."
pip install -r requirements.txt

# 检查安装是否成功
if [ $? -eq 0 ]; then
    echo "依赖安装成功!"
    echo "您现在可以使用以下命令启动 Streamlit 界面:"
    echo "  ./run_streamlit.py"
    echo "或者:"
    echo "  streamlit run streamlit_app.py"
else
    echo "安装过程中出现错误。请检查错误信息并重试。"
    exit 1
fi
