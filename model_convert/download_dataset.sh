#!/bin/bash

# 创建 datasets 目录（如果不存在）
mkdir -p datasets

# 下载到 datasets 目录，并添加错误处理
if ! wget -P datasets https://github.com/AXERA-TECH/SmolVLM2-500M-Video-Instruct.axera/releases/download/calibration/hidden_states.tar; then
    echo "错误：文件下载失败，请手动从以下链接下载："
    echo "https://github.com/AXERA-TECH/SmolVLM2-500M-Video-Instruct.axera/releases/download/calibration/hidden_states.tar"
    exit 1
fi

echo "下载成功！文件保存在 datasets 目录"