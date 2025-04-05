#!/usr/bin/env python
"""
启动 Streamlit 应用的脚本。
"""
import subprocess
import os
import sys
import webbrowser
import time

def main():
    """启动 Streamlit 应用"""
    print("启动 Streamlit 应用...")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Streamlit 应用路径
    streamlit_app = os.path.join(current_dir, "streamlit_app.py")
    
    # 检查 streamlit_app.py 是否存在
    if not os.path.exists(streamlit_app):
        print(f"错误: 找不到 Streamlit 应用文件: {streamlit_app}")
        sys.exit(1)
    
    # 启动 Streamlit
    port = 8501
    url = f"http://localhost:{port}"
    
    print(f"Streamlit 应用将在 {url} 上运行")
    print("正在启动...")
    
    # 启动 Streamlit 进程
    cmd = [sys.executable, "-m", "streamlit", "run", streamlit_app, "--server.port", str(port)]
    
    try:
        # 在新窗口中打开浏览器
        time.sleep(2)  # 给 Streamlit 一些启动时间
        webbrowser.open(url)
        
        # 运行 Streamlit
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStreamlit 应用已停止")
    except Exception as e:
        print(f"启动 Streamlit 应用时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
