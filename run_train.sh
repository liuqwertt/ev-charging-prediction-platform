#!/bin/bash
# 新能源充电桩 - 一键训练脚本 (Ubuntu bash)
# 使用方法: chmod +x run_train.sh && ./run_train.sh

# 严格错误模式：任意命令失败直接退出，等价PowerShell $ErrorActionPreference = "Stop"
set -euo pipefail

# 彩色输出函数（兼容Ubuntu默认终端）
cyan() { echo -e "\033[36m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }
green() { echo -e "\033[32m$1\033[0m"; }
white() { echo -e "\033[37m$1\033[0m"; }

# 进入脚本所在目录（等价 $PSScriptRoot）
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}"

cyan "=============================================="
cyan "新能源充电桩数据预测模型 - 一键训练脚本"
cyan "=============================================="
echo ""

yellow "[1/7] 同步依赖..."
uv sync

echo ""
yellow "[2/7] 下载电池数据..."
uv run python com/neu/Deshdfs.py

echo ""
yellow "[3/7] 下载充电桩数据..."
uv run python com/neu/Nvv2thdfs.py

echo ""
yellow "[4/7] 训练SOC预测模型..."
uv run python ds_battery72_1.py

echo ""
yellow "[5/7] 训练充电时间预测模型..."
uv run python nvv_ds_01_1.py

echo ""
yellow "[6/7] 训练充电费用预测模型..."
uv run python nvv_ds_02_1.py

echo ""
yellow "[7/7] 训练平台选择预测模型..."
uv run python nvv_ds_03_1.py

echo ""
green "=============================================="
green "全部训练完成!"
green "=============================================="
echo ""
white "启动服务命令: uv run python app.py"
white "访问地址: http://localhost:5000"

# 标准正常退出码
exit 0