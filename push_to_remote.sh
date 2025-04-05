#!/bin/bash

# 请将以下 URL 替换为您的实际仓库 URL
REPO_URL="https://github.com/bluegithupu/rag-test.git"

# 关联远程仓库
git remote add origin $REPO_URL

# 将主分支重命名为 main
git branch -M main

# 推送到远程仓库
git push -u origin main

echo "已成功推送到远程仓库！"
