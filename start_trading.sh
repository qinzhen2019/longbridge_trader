#!/bin/bash
cd "$(dirname "$0")"

echo "=========================================="
echo "长桥自动交易程序启动"
echo "时间: $(date)"
echo "使用 caffeinate 防止系统睡眠"
echo "=========================================="

# caffeinate -i: 阻止空闲睡眠
# caffeinate -s: 阻止系统睡眠（需要接电源）
# caffeinate -d: 阻止显示器睡眠

caffeinate -i -s python3 main.py
